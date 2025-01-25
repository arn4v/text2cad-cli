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
const SYSTEM_PROMPT = `You are an expert CAD designer using OpenSCAD. Your task is to generate 3D models based on descriptions and iteratively improve them based on feedback.

CRITICAL: Your responses must exactly follow this format, with no deviations:

<openscad>
// Your OpenSCAD code here
// Use precise measurements
// Follow OpenSCAD best practices
// Consider printability
</openscad>

views:
- name: "front"
  angle: [0, 0, 0]
  distance: 100
- name: "iso"
  angle: [45, 35, 0]
  distance: 100

Do not add any text between sections. Maintain exact indentation as shown.`;

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

    const renders: RenderResult[] = [];

    for (const view of model.views) {
      const outputFile = join(this.tempDir, `${view.name}.png`);

      try {
        // Validate view parameters
        if (!Array.isArray(view.angle) || view.angle.length !== 3) {
          throw new Error(`Invalid angle array for view ${view.name}`);
        }
        if (typeof view.distance !== "number" || view.distance <= 0) {
          throw new Error(`Invalid distance value for view ${view.name}`);
        }

        // Construct camera parameters using eye/center format
        const cameraParams = [
          `0,0,${view.distance}`, // eye position (looking from above)
          "0,0,0", // center position (looking at origin)
        ].join(",");

        const args = [
          "-o",
          outputFile,
          "--colorscheme=Cornfield",
          "--imgsize=1024,768",
          "--viewall",
          "--projection=ortho",
          "--preview",
          "--camera",
          cameraParams,
          modelFile,
        ];

        await execa("openscad", args);

        const imageData = await readFile(outputFile);
        const renderDir = join(RENDERS_DIR, randomUUID());
        await mkdir(renderDir, { recursive: true });

        // Save the rendered image
        const savedImagePath = join(renderDir, `${view.name}.png`);
        await writeFile(savedImagePath, imageData);
        console.log(`Saved render to: ${savedImagePath}`);

        // Add to renders array
        renders.push({
          view: view.name,
          image: imageData.toString("base64"),
        });
      } catch (error: unknown) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to render view ${view.name}: ${errorMessage}`);
      }
    }

    await this.cleanup();
    return renders;
  }

  private async cleanup() {
    if (this.tempDir && existsSync(this.tempDir)) {
      await rm(this.tempDir, { recursive: true, force: true });
      this.tempDir = null;
    }
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
    const codeMatch = response.match(/<openscad>([\s\S]*?)<\/openscad>/);
    if (!codeMatch) {
      console.error("Full response:", response);
      throw new Error(
        "Could not find OpenSCAD code section. Response must include <openscad> tags."
      );
    }

    try {
      const code = codeMatch[1].trim();
      console.log(
        "\nExtracted OpenSCAD code length:",
        code.length,
        "characters"
      );

      // Default views as fallback
      const defaultViews: ViewSpec[] = [
        {
          name: "front",
          angle: [0, 0, 0] as [number, number, number],
          distance: 100,
        },
        {
          name: "iso",
          angle: [45, 35, 0] as [number, number, number],
          distance: 100,
        },
      ];

      let views = defaultViews;
      const viewsSectionMatch = response.match(
        /views:([\s\S]*?)(?=\n\n|<\/|$)/i
      );

      if (viewsSectionMatch) {
        try {
          const viewsText = viewsSectionMatch[1];
          const viewEntries = viewsText.split(/(?:\n\s*)?-\s*/).filter(Boolean);

          views = viewEntries.map((viewText) => {
            const nameMatch = viewText.match(/name\s*:\s*"([^"]+)"/i);
            const angleMatch = viewText.match(/angle\s*:\s*\[([^\]]+)\]/i);
            const distanceMatch = viewText.match(/distance\s*:\s*(\d+)/i);

            if (!nameMatch || !angleMatch) {
              throw new Error("Invalid view format - missing required fields");
            }

            const angleParts = angleMatch[1].split(/\s*,\s*/).map(Number);
            if (angleParts.length !== 3 || angleParts.some(isNaN)) {
              throw new Error(`Invalid angle values: ${angleMatch[1]}`);
            }

            return {
              name: nameMatch[1],
              angle: angleParts as [number, number, number],
              distance: distanceMatch ? parseInt(distanceMatch[1], 10) : 100,
            };
          });

          console.log("Successfully parsed views:", views);
        } catch (error) {
          console.warn("Failed to parse views section, using defaults:", error);
          views = defaultViews;
        }
      }

      return { code, views };
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      throw new Error(
        `Failed to parse response: ${errorMessage}\n\nPlease try the command again.`
      );
    }
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
