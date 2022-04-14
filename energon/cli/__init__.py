import click
import typer
from energon.cli.service import service

app = typer.Typer()

@app.callback()
def callback():
    """
    Typer app, including Click subapp
    """

typer_click_object = typer.main.get_command(app)
typer_click_object.add_command(service, "service")

if __name__ == "__main__":
    typer_click_object()