import { execa } from "execa";
import { Anthropic } from "@anthropic-ai/sdk";
import cac from "cac";
import { writeFile, readFile, mkdir, rm } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { tmpdir, homedir } from "os";
import { randomUUID } from "crypto";

interface ViewSpec {
  name: string;
  angle: [number, number, number];
  distance: number;
}

interface CADModel {
  code: string;
  views: ViewSpec[];
}

interface RenderResult {
  view: string;
  image: string;
}

interface IterationState {
  timestamp: string;
  code: string;
  renders: RenderResult[];
  feedback?: string;
  changes?: string;
}

interface ProjectState {
  id: string;
  originalPrompt: string;
  iterations: IterationState[];
}

type MessageContent =
  | { type: "text"; text: string }
  | {
      type: "image";
      source: { type: "base64"; media_type: "image/png"; data: string };
    };

const SYSTEM_PROMPT = `You are an expert CAD designer using OpenSCAD. Follow these guidelines:

1. Design Principles:
- Use engineering best practices
- Include parametric variables
- Add clearance tolerances (0.2-0.5mm)
- Ensure printability/manufacturability

2. View Requirements:
- Standard engineering views:
  * front: [0, 0, 0] distance: 150
  * top: [90, 0, 0] distance: 200
  * iso: [45, 35, 15] distance: 250
- Minimum 3 views showing key features

3. Response Format:
<openscad>
// Parameterized code here
</openscad>

views:
- name: "front" angle: [0,0,0] distance: 150
- name: "top" angle: [90,0,0] distance: 200
- name: "iso" angle: [45,35,15] distance: 250`;

const STATE_DIR = join(homedir(), ".cad-forge");
const RENDERS_DIR = join(STATE_DIR, "renders");

class OpenSCADRenderer {
  private tempDir: string | null = null;

  async render(model: CADModel): Promise<RenderResult[]> {
    await this.cleanup();
    this.tempDir = join(tmpdir(), `cad-forge-${Date.now()}`);
    await mkdir(this.tempDir, { recursive: true });

    const modelFile = join(this.tempDir, "model.scad");
    await writeFile(modelFile, model.code);

    const renderDir = join(RENDERS_DIR, randomUUID());
    await mkdir(renderDir, { recursive: true });

    const renders: RenderResult[] = [];

    for (const view of model.views) {
      const outputFile = join(this.tempDir, `${view.name}.png`);

      try {
        // Base distance should be closer to the object
        const baseDistance = 100;
        const validatedDistance = Math.max(view.distance, baseDistance);
        const [rotX, rotY, rotZ] = view.angle;

        // Calculate camera position based on the view
        let cameraParams;
        switch (view.name) {
          case "front":
            // Slight angle to show depth, centered on ports
            cameraParams = `0,0,0,0,5,0,${baseDistance}`;
            break;
          case "top":
            // Pure top view with slight rotation for better orientation
            cameraParams = `0,0,0,90,0,10,${baseDistance}`;
            break;
          case "iso":
            // Standard isometric view angles with adjusted distance
            cameraParams = `20,0,0,35,0,25,${baseDistance * 1.2}`;
            break;
          default:
            cameraParams = `0,0,0,${rotX},${rotY},${rotZ},${validatedDistance}`;
        }

        await execa("openscad", [
          "-o",
          outputFile,
          "--imgsize=1024,768",
          "--viewall",
          "--colorscheme=Cornfield",
          "--projection=ortho",
          "--preview",
          `--camera=${cameraParams}`,
          modelFile,
        ]);

        const imageData = await readFile(outputFile);
        const finalPath = join(renderDir, `${view.name}.png`);
        await writeFile(finalPath, imageData);

        renders.push({
          view: view.name,
          image: imageData.toString("base64"),
        });
        console.log(`Rendered ${view.name} view to: ${finalPath}`);
      } catch (error) {
        await this.cleanup();
        throw new Error(
          `Render failed: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }
    }

    await this.cleanup();
    return renders;
  }

  private async cleanup() {
    if (this.tempDir && existsSync(this.tempDir)) {
      await rm(this.tempDir, { recursive: true, force: true });
    }
    this.tempDir = null;
  }
}

class CADSession {
  private state: ProjectState | null = null;
  private anthropic: Anthropic;
  private stateFile: string;

  constructor(apiKey: string) {
    this.anthropic = new Anthropic({ apiKey });
    this.stateFile = join(STATE_DIR, "state.json");
  }

  async generateModel(prompt: string): Promise<CADModel> {
    this.state = {
      id: randomUUID(),
      originalPrompt: prompt,
      iterations: [],
    };
    const model = await this.createModel({
      role: "user",
      content: prompt,
    });
    await this.addIteration(model);
    return model;
  }

  async iterateModel(feedback: string): Promise<CADModel> {
    await this.loadState();
    if (!this.state?.iterations.length) {
      throw new Error("No previous model found");
    }

    const lastIteration =
      this.state.iterations[this.state.iterations.length - 1];
    const model = await this.createModel({
      role: "user",
      content: this.formatIterationMessage(feedback, lastIteration.renders),
    });

    await this.addIteration(model, feedback);
    return model;
  }

  private async createModel(message: {
    role: "user";
    content: string | MessageContent[];
  }): Promise<CADModel> {
    const response = await this.anthropic.messages.create({
      model: "claude-3-5-sonnet-latest",
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [message],
      stream: true,
    });

    let fullResponse = "";
    for await (const chunk of response) {
      if (
        chunk.type === "content_block_delta" &&
        chunk.delta.type === "text_delta"
      ) {
        process.stdout.write(chunk.delta.text);
        fullResponse += chunk.delta.text;
      }
    }
    console.log();
    return this.parseResponse(fullResponse);
  }

  private parseResponse(response: string): CADModel {
    const codeMatch = response.match(/<openscad>([\s\S]*?)<\/openscad>/i);
    if (!codeMatch) throw new Error("Missing OpenSCAD code section");

    const views: ViewSpec[] = [];
    const viewSection = response.match(/views:([\s\S]*?)(?=\n\n|<\/|$)/i)?.[1];

    if (viewSection) {
      const viewEntries = viewSection.split(/-\s*/).slice(1);
      for (const entry of viewEntries) {
        try {
          const nameMatch = entry.match(/name\s*:\s*"([^"]+)"/i);
          const angleMatch = entry.match(/angle\s*:\s*\[([^\]]+)\]/i);
          const distanceMatch = entry.match(/distance\s*:\s*(\d+)/i);

          if (!nameMatch?.[1] || !angleMatch?.[1]) continue;

          const angleParts = angleMatch[1]
            .split(/\s*,\s*/)
            .map(Number)
            .filter((n) => !isNaN(n));

          if (angleParts.length !== 3) continue;

          views.push({
            name: nameMatch[1].toLowerCase(),
            angle: angleParts as [number, number, number],
            distance: distanceMatch ? parseInt(distanceMatch[1]) : 150,
          });
        } catch (error) {
          console.warn("Skipping invalid view entry:", error);
        }
      }
    }

    return {
      code: codeMatch[1].trim(),
      views:
        views.length >= 3
          ? views
          : [
              { name: "front", angle: [0, 0, 0], distance: 150 },
              { name: "top", angle: [90, 0, 0], distance: 200 },
              { name: "iso", angle: [45, 35, 15], distance: 250 },
            ],
    };
  }

  private formatIterationMessage(
    feedback: string,
    renders: RenderResult[]
  ): MessageContent[] {
    if (!this.state) throw new Error("No active session");

    const content: MessageContent[] = [
      {
        type: "text",
        text: `Original Request: ${
          this.state.originalPrompt
        }\n\nFeedback: ${feedback}\n\nCurrent Code:\n${
          this.state.iterations[this.state.iterations.length - 1].code
        }\n\nPlease analyze these renders:`,
      },
    ];

    renders.forEach((render) => {
      content.push(
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: render.image,
          },
        },
        {
          type: "text",
          text: `${render.view} view`,
        }
      );
    });

    content.push({
      type: "text",
      text: "Provide improved OpenSCAD code maintaining the required format.",
    });

    return content;
  }

  private async addIteration(model: CADModel, feedback?: string) {
    if (!this.state) throw new Error("No active session");
    this.state.iterations.push({
      timestamp: new Date().toISOString(),
      code: model.code,
      renders: [],
      feedback,
    });
    await this.saveState();
  }

  private async loadState() {
    try {
      const data = await readFile(this.stateFile, "utf8");
      this.state = JSON.parse(data);
    } catch (error) {
      throw new Error("No existing session found");
    }
  }

  private async saveState() {
    if (!this.state) throw new Error("No active session");
    await mkdir(STATE_DIR, { recursive: true });
    await writeFile(this.stateFile, JSON.stringify(this.state, null, 2));
  }
}

async function main() {
  const cli = cac("cad-forge");

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("ANTHROPIC_API_KEY environment variable required");
    process.exit(1);
  }

  try {
    await execa("openscad", ["--version"]);
  } catch {
    console.error("OpenSCAD not found in PATH");
    process.exit(1);
  }

  await mkdir(RENDERS_DIR, { recursive: true });
  const session = new CADSession(apiKey);
  const renderer = new OpenSCADRenderer();

  cli
    .command("create <prompt>", "Create new CAD design")
    .action(async (prompt: string) => {
      try {
        console.log("Generating model...");
        const model = await session.generateModel(prompt);
        console.log("Rendering views...");
        const renders = await renderer.render(model);

        const state = await readFile(join(STATE_DIR, "state.json"), "utf8");
        const projectState: ProjectState = JSON.parse(state);
        projectState.iterations[projectState.iterations.length - 1].renders =
          renders;
        await writeFile(
          join(STATE_DIR, "state.json"),
          JSON.stringify(projectState, null, 2)
        );

        console.log("\nModel created successfully!");
        console.log("Renders saved to:", RENDERS_DIR);
      } catch (error) {
        console.error(
          "Error:",
          error instanceof Error ? error.message : String(error)
        );
        process.exit(1);
      }
    });

  cli
    .command("iterate <feedback>", "Improve existing design")
    .action(async (feedback: string) => {
      try {
        console.log("Processing feedback...");
        const model = await session.iterateModel(feedback);
        console.log("Rendering new views...");
        const renders = await renderer.render(model);

        const state = await readFile(join(STATE_DIR, "state.json"), "utf8");
        const projectState: ProjectState = JSON.parse(state);
        projectState.iterations[projectState.iterations.length - 1].renders =
          renders;
        await writeFile(
          join(STATE_DIR, "state.json"),
          JSON.stringify(projectState, null, 2)
        );

        console.log("\nDesign updated successfully!");
        console.log("Updated renders saved to:", RENDERS_DIR);
      } catch (error) {
        console.error(
          "Error:",
          error instanceof Error ? error.message : String(error)
        );
        process.exit(1);
      }
    });

  cli.help();
  cli.parse();
}

main().catch(console.error);
