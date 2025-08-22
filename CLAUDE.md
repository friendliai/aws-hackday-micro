# Claude Code Project Configuration

This project is configured with automatic Python formatting and linting using ruff, TOML formatting
using taplo, and Markdown formatting using mdformat.

## Available Commands

### Python Linting and Formatting

- `make lint` - Check for linting issues in src/ and examples/ directories
- `make fix` - Automatically fix linting issues and format code in src/ and examples/
- `make format` - Format all Python files in src/ and examples/
- `ruff check src/ examples/` - Check for linting issues
- `ruff check --fix src/ examples/` - Automatically fix linting issues
- `ruff format src/ examples/` - Format Python files
- `ruff check --select I --fix src/ examples/` - Fix import sorting only

### TOML Formatting

- `taplo format **/*.toml` - Format all TOML files
- `taplo check **/*.toml` - Check TOML files for formatting issues
- `taplo format pyproject.toml` - Format specific TOML file

### Markdown Formatting

- `make format` - Format all Markdown, Python, and TOML files
- `find . -name "*.md" | xargs uv run mdformat --wrap 100` - Format all Markdown files
- `uv run mdformat --wrap 100 README.md` - Format specific Markdown file

### Running Tests

- `pytest` - Run all tests
- `pytest -v` - Run tests with verbose output
- `pytest -x` - Stop on first failure

### Type Checking

- `mypy src/ examples/` - Run type checking on source code and examples

## Automatic Formatting

This project has Claude Code hooks configured to automatically:

1. Format Python files with ruff after every edit
1. Fix common linting issues automatically
1. Format TOML files with taplo after every edit
1. Format Markdown files with mdformat after every edit
1. Report any remaining linting issues that need manual intervention

## Code Style

### Python

- Line length: 100 characters
- Quote style: Double quotes
- Import sorting: Automatic with isort rules
- Python version: 3.13

### TOML

- Line length: 100 characters
- Array trailing commas: Enabled
- Array auto-expand: Enabled for readability
- Key reordering: Disabled to preserve structure
- Aligned entries: Disabled for consistency

### Markdown

- Line length: 100 characters
- GitHub Flavored Markdown (GFM) support
- Table formatting enabled
- Preserves original structure and content

## Important Notes

- All Python files in src/ and examples/ are automatically formatted when edited through Claude Code
- All TOML files are automatically formatted when edited through Claude Code
- All Markdown files are automatically formatted when edited through Claude Code
- The formatters run after Edit, MultiEdit, and Write operations
- Linting issues that can be auto-fixed will be fixed automatically
- Remaining issues will be reported in the output

## VSCode Integration

- Install the "Even Better TOML" extension for TOML formatting in VSCode
- Install the "markdownlint" extension for Markdown formatting and linting in VSCode
- Settings are pre-configured to match Claude Code formatting
- Format on save is enabled for Python, TOML, and Markdown files
