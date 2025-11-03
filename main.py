import asyncio
import argparse
import os

from app.bot.bot_app import run_bot


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Gipotez runner")
    parser.add_argument("--mode", choices=["tg", "api"], default="tg", help="Run Telegram bot or FastAPI server")
    parser.add_argument("--host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    args = parser.parse_args()

    if args.mode == "tg":
        asyncio.run(run_bot())
    else:
        # FastAPI via uvicorn
        import uvicorn
        uvicorn.run("app.api.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
