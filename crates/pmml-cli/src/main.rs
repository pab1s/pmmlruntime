use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pmml-runtime", version, about = "PMML 4.4 runtime — ONNX-style", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect PMML file (prints DataDictionary, MiningSchema)
    Inspect {
        #[arg(long)]
        model: String,
    },
    /// Run scoring
    Run {
        #[arg(long)]
        model: String,
        #[arg(long)]
        input: Option<String>,
        #[arg(long)]
        output: Option<String>,
    },
    Verify {
        #[arg(long)]
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { model } => {
            println!("inspect: {model} (stub v0)");
        }
        Commands::Run { model, input, output } => {
            println!("run: model={model} input={input:?} output={output:?} (stub v0)");
        }
        Commands::Verify { model } => {
            println!("verify: {model} (stub v0)");
        }
    }
    Ok(())
}
