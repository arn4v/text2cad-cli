import { execa } from "execa";
import { Anthropic } from "@anthropic-ai/sdk";
import cac from "cac";
import { parse } from "yaml";
import { writeFile, readFile, mkdir, rm } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { tmpdir, homedir } from "os";
import { randomUUID } from "crypto";
import { config } from "dotenv";

// Load environment variables from .env file
config();

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
        await execa("openscad", [
          "-o",
          outputFile,
          "--camera",
          `${view.angle.join(",")},${view.distance}`,
          "--preview",
          modelFile,
        ]);

        const imageData = await readFile(outputFile);
        const renderDir = join(RENDERS_DIR, randomUUID());
        await mkdir(renderDir, { recursive: true });
        await writeFile(join(renderDir, `${view.name}.png`), imageData);

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
      model: "claude-3-opus-20240229",
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
    for await (const chunk of response) {
      if (
        chunk.type === "content_block_delta" &&
        chunk.delta.type === "text_delta"
      ) {
        const text = chunk.delta.text;
        fullResponse += text;
        process.stdout.write(text);
      }
    }
    console.log("\n"); // Add newline after streaming completes

    return this.parseResponse(fullResponse);
  }

  private parseResponse(response: string): CADModel {
    // Get OpenSCAD code
    const codeMatch = response.match(/<openscad>([\s\S]*?)<\/openscad>/);
    if (!codeMatch) {
      throw new Error(
        "Could not find OpenSCAD code section. Response must include <openscad> tags."
      );
    }

    // Get YAML section with stricter matching
    const yamlMatch = response.match(
      /views:\s*((?:[\s\S]*?\n  - [\s\S]*?)+)(?:\n\n|$)/
    );
    if (!yamlMatch) {
      throw new Error("Could not find properly formatted views section");
    }

    try {
      const code = codeMatch[1].trim();
      // Ensure proper YAML indentation
      const yamlText =
        "views:\n" +
        yamlMatch[1]
          .split("\n")
          .map((line) => `  ${line}`)
          .join("\n");
      const parsedYaml = parse(yamlText);

      if (!parsedYaml || !Array.isArray(parsedYaml.views)) {
        throw new Error("Invalid views format - must be an array");
      }

      // Validate view structure
      parsedYaml.views.forEach((view: any, index: number) => {
        if (!view.name || typeof view.name !== "string") {
          throw new Error(`View ${index} missing name property`);
        }
        if (!Array.isArray(view.angle) || view.angle.length !== 3) {
          throw new Error(
            `View ${index} (${view.name}) has invalid angle format`
          );
        }
        if (typeof view.distance !== "number") {
          throw new Error(
            `View ${index} (${view.name}) missing distance property`
          );
        }
      });

      return { code, views: parsedYaml.views };
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

    // Add each render with its view description
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
      renders: [], // Filled after rendering
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

  // Check environment
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error(`Error: ANTHROPIC_API_KEY environment variable is required.
Create a .env file in the project root with:
ANTHROPIC_API_KEY=your-api-key-here`);
    process.exit(1);
  }

  // Check OpenSCAD installation
  try {
    await execa("openscad", ["--version"]);
  } catch (error) {
    console.error("Error: OpenSCAD must be installed and available in PATH");
    process.exit(1);
  }

  // Create required directories
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
        console.log(model);

        console.log("Rendering views...");
        const renders = await renderer.render(model);

        // Update the iteration with renders
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

        // Update the iteration with renders
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
