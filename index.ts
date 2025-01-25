import { execa } from "execa";
import { Anthropic } from "@anthropic-ai/sdk";
import cac from "cac";
import { writeFile, readFile, mkdir, rm } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { tmpdir, homedir } from "os";
import { randomUUID } from "crypto";

// Types
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
  image: string; // base64
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

interface ImageContent {
  type: "image";
  source: {
    type: "base64";
    media_type: "image/png";
    data: string;
  };
}

interface TextContent {
  type: "text";
  text: string;
}

type MessageContent = TextContent | ImageContent;

// Constants
const SYSTEM_PROMPT = `You are an expert CAD designer using OpenSCAD. Follow these strict rules:

1. VIEW ORIENTATION:
- Use standard engineering views with RIGHT-HAND RULE coordinate system:
  * +X = Right, +Y = Back, +Z = Up
  * Front view looks along +Y axis
  * Top view looks along +Z axis
- Include these required views:
  * front: [0, 0, 0] (facing forward)
  * back: [0, 180, 0]
  * left: [0, 90, 0]
  * right: [0, -90, 0]
  * top: [90, 0, 0]
  * bottom: [-90, 0, 0]
  * iso: [45, 35, 0]

2. COMPONENT PLACEMENT:
- Ports/connectors must be on correct faces:
  * USB/HDMI on front/back as per actual device
  * Mounting holes on bottom
  * Ventilation on top/sides
- Verify component orientation with 3D coordinate system

3. DESIGN RULES:
- Model origin at center of base
- Align components with axes
- Add comments for complex geometry

4. RESPONSE FORMAT:
<openscad>
// Code here
</openscad>

views:
- name: "front" angle: [0,0,0] distance: 150
- name: "back" angle: [0,180,0] distance: 150
- name: "left" angle: [0,90,0] distance: 150
- name: "right" angle: [0,-90,0] distance: 150 
- name: "top" angle: [90,0,0] distance: 150
- name: "bottom" angle: [-90,0,0] distance: 150
- name: "iso" angle: [45,35,0] distance: 200`;

const STATE_DIR = join(homedir(), ".cad-forge");
const RENDERS_DIR = join(STATE_DIR, "renders");

// OpenSCAD Renderer
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
        // Convert angles to radians
        const azimuth = (view.angle[0] * Math.PI) / 180;
        const elevation = (view.angle[1] * Math.PI) / 180;
        const tilt = (view.angle[2] * Math.PI) / 180;

        // Calculate eye position using spherical coordinates
        const eye = [
          view.distance * Math.cos(azimuth) * Math.cos(elevation),
          view.distance * Math.sin(azimuth) * Math.cos(elevation),
          view.distance * Math.sin(elevation),
        ];

        // Calculate center point with tilt adjustment
        const center = [
          Math.cos(tilt) * 0.1 * view.distance,
          Math.sin(tilt) * 0.1 * view.distance,
          0,
        ];

        const cameraParams = `=${eye.join(",")},${center.join(",")}`;

        await execa("openscad", [
          "-o",
          outputFile,
          "--camera",
          cameraParams,
          "--viewall",
          "--imgsize=1024,768",
          "--colorscheme=Cornfield",
          "--projection=ortho",
          "--preview",
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
          `Failed to render ${view.name}: ${
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
// LLM Session Handler
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
      content: this.formatInitialPrompt(prompt),
    });
    await this.addIteration(model);
    return model;
  }

  async iterateModel(feedback: string): Promise<CADModel> {
    await this.loadState();

    if (!this.state || this.state.iterations.length === 0) {
      throw new Error('No previous model found. Use "create" first.');
    }

    const lastIteration =
      this.state.iterations[this.state.iterations.length - 1];
    if (!lastIteration.renders || lastIteration.renders.length === 0) {
      throw new Error("No renders found from previous iteration");
    }

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
      messages: [
        {
          role: "user",
          content: message.content,
        },
      ] satisfies Anthropic.Messages.MessageParam[],
      stream: true,
    });

    let fullResponse = "";
    let lastChunkWasNewline = false;

    for await (const chunk of response) {
      if (
        chunk.type === "content_block_delta" &&
        chunk.delta.type === "text_delta"
      ) {
        const text = chunk.delta.text;
        if (text === "\n") {
          if (!lastChunkWasNewline) {
            process.stdout.write(text);
            lastChunkWasNewline = true;
          }
        } else {
          process.stdout.write(text);
          lastChunkWasNewline = false;
        }
        fullResponse += text;
      }
    }

    if (!lastChunkWasNewline) {
      console.log();
    }

    return this.parseResponse(fullResponse);
  }

  private parseResponse(response: string): CADModel {
    const codeMatch = response.match(/<openscad>([\s\S]*?)<\/openscad>/i);
    if (!codeMatch) throw new Error("Missing OpenSCAD code section");

    const code = codeMatch[1].trim();
    const views: ViewSpec[] = [];
    const viewErrors: string[] = [];

    const viewsSection = response.match(/views:([\s\S]*?)(?=\n\n|<\/|$)/i)?.[1];
    if (viewsSection) {
      const viewEntries = viewsSection.split(/(?:\n\s*)?-\s*/).filter(Boolean);

      for (const [index, entry] of viewEntries.entries()) {
        try {
          // Flexible parsing with error handling
          const nameMatch = entry.match(/name\s*[:=]\s*["']?([\w-]+)["']?/i);
          const angleMatch = entry.match(
            /angle\s*[:=]\s*\[?\s*([-\d\s.,]+)\s*\]?/i
          );
          const distanceMatch = entry.match(/distance\s*[:=]\s*(\d+)/i);

          if (!nameMatch?.[1])
            throw new Error(`Missing name in view ${index + 1}`);
          if (!angleMatch?.[1])
            throw new Error(`Missing angles in view ${index + 1}`);

          const angleParts = angleMatch[1]
            .split(/,\s*|\s+/)
            .map(Number)
            .filter((n) => !isNaN(n));

          if (angleParts.length !== 3)
            throw new Error(`Invalid angles in view ${index + 1}`);

          views.push({
            name: nameMatch[1].toLowerCase(),
            angle: angleParts as [number, number, number],
            distance: distanceMatch ? parseInt(distanceMatch[1], 10) : 150,
          });
        } catch (error) {
          viewErrors.push(
            `View ${index + 1}: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
        }
      }

      if (viewErrors.length > 0) {
        console.warn("View parsing issues:\n" + viewErrors.join("\n"));
      }
    }

    // Default views if parsing failed
    const defaultViews: ViewSpec[] = [
      { name: "front", angle: [0, 0, 0], distance: 150 },
      { name: "back", angle: [180, 0, 0], distance: 150 },
      { name: "left", angle: [90, 0, 0], distance: 150 },
      { name: "right", angle: [-90, 0, 0], distance: 150 },
      { name: "top", angle: [0, 90, 0], distance: 200 },
      { name: "bottom", angle: [0, -90, 0], distance: 200 },
      { name: "iso", angle: [45, 35, 15], distance: 300 },
    ];

    return {
      code,
      views: views.length > 3 ? views : defaultViews, // Only use parsed views if most are valid
    };
  }
  private formatInitialPrompt(prompt: string): string {
    return `Design a 3D model based on this description:
${prompt}

IMPORTANT: Your response must follow this exact format:

<openscad>
// OpenSCAD code here
</openscad>

views:
- name: "front"
  angle: [0, 0, 0]
  distance: 100
- name: "iso"
  angle: [45, 35, 0]
  distance: 100

Do not include any other text between these sections.`;
  }

  private formatIterationMessage(
    feedback: string,
    renders: RenderResult[]
  ): MessageContent[] {
    if (!this.state) throw new Error("No active session");

    const latestCode =
      this.state.iterations[this.state.iterations.length - 1].code;

    const content: MessageContent[] = [
      {
        type: "text",
        text: `Original request: ${this.state.originalPrompt}

Previous OpenSCAD code:
${latestCode}

User feedback:
${feedback}

Please analyze the following renders and make the requested improvements:`,
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
          text: `↑ ${render.view} view`,
        }
      );
    });

    content.push({
      type: "text",
      text: "Provide updated OpenSCAD code and view specifications following the standard format.",
    });

    return content;
  }

  private async addIteration(model: CADModel, feedback?: string) {
    if (!this.state) throw new Error("No active session");

    const iteration: IterationState = {
      timestamp: new Date().toISOString(),
      code: model.code,
      renders: [],
      feedback,
    };

    this.state.iterations.push(iteration);
    await this.saveState();
  }

  private async loadState() {
    try {
      const data = await readFile(this.stateFile, "utf8");
      this.state = JSON.parse(data);
    } catch (error) {
      throw new Error('No existing CAD session found. Use "create" first.');
    }
  }

  private async saveState() {
    if (!this.state) throw new Error("No active session");

    await mkdir(STATE_DIR, { recursive: true });
    await writeFile(this.stateFile, JSON.stringify(this.state, null, 2));
  }
}

// CLI Setup
async function main() {
  const cli = cac("cad-forge");

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error(`Error: ANTHROPIC_API_KEY environment variable is required.
Create a .env file in the project root with:
ANTHROPIC_API_KEY=your-api-key-here`);
    process.exit(1);
  }

  try {
    await execa("openscad", ["--version"]);
  } catch (error) {
    console.error("Error: OpenSCAD must be installed and available in PATH");
    process.exit(1);
  }

  await mkdir(STATE_DIR, { recursive: true });
  await mkdir(RENDERS_DIR, { recursive: true });

  const session = new CADSession(apiKey);
  const renderer = new OpenSCADRenderer();

  cli
    .command("create <prompt>", "Create a new CAD model")
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
        console.log("\nOpenSCAD code:");
        console.log(model.code);
        console.log("\nRenders saved to:", RENDERS_DIR);
      } catch (error: unknown) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        console.error("Error:", errorMessage);
        process.exit(1);
      }
    });

  cli
    .command("iterate <feedback>", "Improve existing model based on feedback")
    .action(async (feedback: string) => {
      try {
        console.log("Loading previous model...");
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

        console.log("\nModel updated successfully!");
        console.log("\nUpdated OpenSCAD code:");
        console.log(model.code);
        console.log("\nRenders saved to:", RENDERS_DIR);
      } catch (error: unknown) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        console.error("Error:", errorMessage);
        process.exit(1);
      }
    });

  cli.help();
  cli.parse();
}

main().catch(console.error);
