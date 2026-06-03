"""Command-line interface for doc-agent."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from .agent import DocAgent

console = Console()
err_console = Console(stderr=True)


@click.group()
@click.version_option()
def cli() -> None:
    """doc-agent — Auto-generate and sync API documentation with source code."""


# ── generate ──────────────────────────────────────────────────────────────────


@cli.command("generate")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where documentation files will be written.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "rst"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Output format for generated documentation.",
)
@click.option(
    "--model",
    default=DocAgent.MODEL,
    show_default=True,
    help="Claude model to use for generation.",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    help="Anthropic API key (or set ANTHROPIC_API_KEY env var).",
)
def generate(
    src_dir: Path,
    output_dir: Path,
    fmt: str,
    model: str,
    api_key: str | None,
) -> None:
    """Generate documentation for Python source files in SRC_DIR.

    SRC_DIR  Root directory containing Python source files to document.
    """
    if not api_key:
        err_console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY is not set. "
            "Pass --api-key or export the environment variable."
        )
        sys.exit(1)

    agent = DocAgent(api_key=api_key, model=model)

    console.print(
        Panel.fit(
            f"[bold cyan]doc-agent[/bold cyan]  generate\n"
            f"  src  : [green]{src_dir}[/green]\n"
            f"  out  : [green]{output_dir}[/green]\n"
            f"  fmt  : {fmt}\n"
            f"  model: {model}",
            title="Configuration",
        )
    )

    with console.status("[bold green]Generating documentation…"):
        try:
            written = agent.generate(src_dir, output_dir, fmt=fmt)
        except anthropic_import_error() as exc:  # type: ignore[misc]
            err_console.print(f"[red]API error:[/red] {exc}")
            sys.exit(1)

    if not written:
        console.print("[yellow]No public API found — nothing was written.[/yellow]")
        return

    table = Table(title="Generated Documentation", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    for path in written:
        size = path.stat().st_size
        table.add_row(str(path), f"{size:,} bytes")

    console.print(table)
    console.print(f"\n[bold green]Done.[/bold green] Wrote {len(written)} file(s).")


# ── check ──────────────────────────────────────────────────────────────────────


@cli.command("check")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory containing existing documentation to check.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "rst"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Format of the existing documentation.",
)
@click.option(
    "--model",
    default=DocAgent.MODEL,
    show_default=True,
    help="Claude model to use for drift checking.",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    help="Anthropic API key (or set ANTHROPIC_API_KEY env var).",
)
def check(
    src_dir: Path,
    output_dir: Path,
    fmt: str,
    model: str,
    api_key: str | None,
) -> None:
    """Check whether documentation in OUTPUT_DIR is in sync with SRC_DIR.

    Exits with code 1 when any drift is detected (suitable for CI pipelines).

    SRC_DIR  Root directory containing Python source files.
    """
    if not api_key:
        err_console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY is not set. "
            "Pass --api-key or export the environment variable."
        )
        sys.exit(1)

    agent = DocAgent(api_key=api_key, model=model)

    console.print(
        Panel.fit(
            f"[bold cyan]doc-agent[/bold cyan]  check\n"
            f"  src  : [green]{src_dir}[/green]\n"
            f"  docs : [green]{output_dir}[/green]\n"
            f"  fmt  : {fmt}\n"
            f"  model: {model}",
            title="Configuration",
        )
    )

    with console.status("[bold yellow]Checking for drift…"):
        drifted = agent.check(src_dir, output_dir, fmt=fmt)

    if not drifted:
        console.print("[bold green]All documentation is up to date.[/bold green]")
        sys.exit(0)

    table = Table(
        title="Documentation Drift Detected",
        show_header=True,
        style="red",
    )
    table.add_column("Documentation File", style="cyan")
    table.add_column("Reason", style="yellow")
    for path, reason in drifted:
        table.add_row(str(path), reason)

    console.print(table)
    err_console.print(
        f"\n[bold red]Drift detected in {len(drifted)} file(s). "
        "Re-run `doc-agent generate` to update.[/bold red]"
    )
    sys.exit(1)


def anthropic_import_error():
    """Return Anthropic API errors as a tuple for except clauses."""
    try:
        import anthropic  # noqa: PLC0415
        return (anthropic.APIError,)
    except ImportError:
        return (Exception,)


def main() -> None:
    """Entry-point for the ``doc-agent`` command."""
    cli()


if __name__ == "__main__":
    main()
