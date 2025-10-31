# Assert uv exists
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_NC='\033[0m' # No Color
# if linux, set
if [[ "$(uname)" -eq "darwin" ]]; then
  alias CECHO='echo'
else
  alias CECHO='echo -e'
fi

command -v uv >/dev/null 2>&1 || {
  CECHO "${COLOR_RED}uv is not installed.${COLOR_NC}"
  exit 1
}

# Assert Im in the git Repo
if [ ! -d .git ]; then
  CECHO "${COLOR_RED}You are not in root of the git repository.${COLOR_NC}"
  exit 1
fi

# Test if .venv exists
if [ ! -d .venv ]; then
  CECHO "${COLOR_YELLOW}Creating virtual environment...${COLOR_NC}"
  uv venv
fi

source .venv/bin/activate>/dev/null 2>&1

if [ $? -eq 0 ]; then
  CECHO "${COLOR_GREEN}==> Virtual environment activated.${COLOR_NC}"
else
  CECHO "${COLOR_RED}Failed to activate virtual environment.${COLOR_NC}"
  exit 1
fi

uv sync --inexact > /dev/null

if [ $? -eq 0 ]; then
  CECHO "${COLOR_GREEN}==> Dependencies installed.${COLOR_NC}"
else
  CECHO "${COLOR_RED}Failed to install dependencies.${COLOR_NC}"
  exit 1
fi


uv pip list | grep pymathprim | count 1 > /dev/null 2>&1
if [ ! $? -eq 0 ]; then
  CECHO "${COLOR_YELLOW}pymathprim is not installed, some functionality will not work...${COLOR_NC}"
else
  CECHO "${COLOR_GREEN}==> current pymathprim version: $(uv run python -c "import pymathprim; print(pymathprim.libpymathprim.__version__)")${COLOR_NC}"
fi

CECHO "${COLOR_GREEN}Done.${COLOR_NC}"

