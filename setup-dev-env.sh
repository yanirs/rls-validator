#!/bin/bash
set -e

if [ "$1" != "--yes" ]; then
  echo "Warning: This script should run in a disposable machine or container. Do you want to continue?"
  read -p "Are you sure? [y/N] " response
  if [ "${response^}" != "Y" ]; then
      exit
  fi
fi
pipx install poetry==$(cat .poetry-version)
poetry install
poetry run pre-commit install
