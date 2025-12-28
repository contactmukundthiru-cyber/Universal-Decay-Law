#!/usr/bin/env python3
"""
Universal Decay Law - Application Runner.

Entry point for running the FastAPI server and initializing the database.
All data collection happens through the API/dashboard - NO synthetic data.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


async def init_database():
    """Initialize the database schema."""
    from database.connection import init_db
    await init_db()
    print("Database initialized successfully.")


async def collect_reddit_data(subreddit: str, limit: int, output_dir: str):
    """
    Collect real engagement data from Reddit.

    Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env
    """
    from config.settings import get_settings
    from src.data.reddit import RedditConnector
    import json

    settings = get_settings()

    if not settings.reddit_client_id or not settings.reddit_client_secret:
        print("ERROR: Reddit API credentials not configured.")
        print("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
        return

    print(f"Collecting data from r/{subreddit}...")

    connector = RedditConnector(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent or "DecayLawResearch/1.0"
    )

    users_data = await connector.collect_users(
        subreddit=subreddit,
        limit=limit
    )

    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"reddit_{subreddit}_{len(users_data)}_users.json"

    # Serialize user data
    serialized = []
    for user in users_data:
        serialized.append({
            "user_id": user.user_id,
            "platform": user.platform,
            "time_array": user.time_array,
            "engagement_array": user.engagement_array,
            "activities_count": len(user.activities),
            "date_range": {
                "start": user.date_range[0].isoformat() if user.date_range else None,
                "end": user.date_range[1].isoformat() if user.date_range else None,
            },
            "metadata": user.metadata
        })

    with open(output_file, 'w') as f:
        json.dump(serialized, f, indent=2)

    print(f"Collected {len(users_data)} users")
    print(f"Data saved to: {output_file}")


async def collect_github_data(query: str, limit: int, output_dir: str):
    """
    Collect real engagement data from GitHub.

    Requires GITHUB_TOKEN in .env
    """
    from config.settings import get_settings
    from src.data.github import GitHubConnector
    import json

    settings = get_settings()

    if not settings.github_token:
        print("ERROR: GitHub token not configured.")
        print("Please set GITHUB_TOKEN in .env")
        return

    print(f"Collecting GitHub data (query: {query})...")

    connector = GitHubConnector(token=settings.github_token)

    users_data = await connector.collect_users(
        query=query,
        limit=limit
    )

    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"github_{len(users_data)}_users.json"

    # Serialize user data
    serialized = []
    for user in users_data:
        serialized.append({
            "user_id": user.user_id,
            "platform": user.platform,
            "time_array": user.time_array,
            "engagement_array": user.engagement_array,
            "activities_count": len(user.activities),
            "date_range": {
                "start": user.date_range[0].isoformat() if user.date_range else None,
                "end": user.date_range[1].isoformat() if user.date_range else None,
            },
            "metadata": user.metadata
        })

    with open(output_file, 'w') as f:
        json.dump(serialized, f, indent=2)

    print(f"Collected {len(users_data)} users")
    print(f"Data saved to: {output_file}")


async def collect_wikipedia_data(limit: int, output_dir: str):
    """
    Collect real engagement data from Wikipedia.

    No API key required.
    """
    from src.data.wikipedia import WikipediaConnector
    import json

    print("Collecting Wikipedia editor data...")

    connector = WikipediaConnector()

    users_data = await connector.collect_users(limit=limit)

    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"wikipedia_{len(users_data)}_users.json"

    # Serialize user data
    serialized = []
    for user in users_data:
        serialized.append({
            "user_id": user.user_id,
            "platform": user.platform,
            "time_array": user.time_array,
            "engagement_array": user.engagement_array,
            "activities_count": len(user.activities),
            "date_range": {
                "start": user.date_range[0].isoformat() if user.date_range else None,
                "end": user.date_range[1].isoformat() if user.date_range else None,
            },
            "metadata": user.metadata
        })

    with open(output_file, 'w') as f:
        json.dump(serialized, f, indent=2)

    print(f"Collected {len(users_data)} users")
    print(f"Data saved to: {output_file}")


def check_env():
    """Check environment configuration and report status."""
    from config.settings import get_settings

    print("=" * 60)
    print("UNIVERSAL DECAY LAW - ENVIRONMENT CHECK")
    print("=" * 60)

    settings = get_settings()

    print("\n[Database]")
    db_url = settings.database.url
    print(f"  URL: {db_url[:50]}..." if len(db_url) > 50 else f"  URL: {db_url}")

    print("\n[API Credentials]")

    # Reddit
    if settings.api_keys.reddit_client_id and settings.api_keys.reddit_client_secret:
        print("  Reddit: CONFIGURED")
    else:
        print("  Reddit: NOT CONFIGURED (set API_REDDIT_CLIENT_ID, API_REDDIT_CLIENT_SECRET)")

    # GitHub
    if settings.api_keys.github_token:
        print("  GitHub: CONFIGURED")
    else:
        print("  GitHub: NOT CONFIGURED (set API_GITHUB_TOKEN)")

    # Strava
    if settings.api_keys.strava_client_id and settings.api_keys.strava_client_secret:
        print("  Strava: CONFIGURED")
    else:
        print("  Strava: NOT CONFIGURED (set API_STRAVA_CLIENT_ID, API_STRAVA_CLIENT_SECRET)")

    # Last.fm
    if settings.api_keys.lastfm_api_key:
        print("  Last.fm: CONFIGURED")
    else:
        print("  Last.fm: NOT CONFIGURED (set API_LASTFM_API_KEY)")

    # Wikipedia (no key needed)
    print("  Wikipedia: AVAILABLE (no API key required)")

    print("\n[Usage]")
    print("  1. Configure API credentials in .env file")
    print("  2. Run: python3 run.py init-db")
    print("  3. Run: python3 run.py server")
    print("  4. Open dashboard at http://localhost:3000")
    print("  5. Use dashboard to collect data and run analyses")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Decay Law of Human Engagement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run.py check-env                        # Check configuration
  python3 run.py init-db                          # Initialize database
  python3 run.py server                           # Run API server
  python3 run.py server --reload                  # Run with auto-reload

  # Data collection (configure .env first):
  python3 run.py collect reddit --subreddit python --limit 100
  python3 run.py collect github --query "machine learning" --limit 50
  python3 run.py collect wikipedia --limit 100
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check environment
    subparsers.add_parser("check-env", help="Check environment configuration")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run the FastAPI server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Init database command
    subparsers.add_parser("init-db", help="Initialize the database")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect real data from platforms")
    collect_subparsers = collect_parser.add_subparsers(dest="platform", help="Platform to collect from")

    # Reddit collection
    reddit_parser = collect_subparsers.add_parser("reddit", help="Collect from Reddit")
    reddit_parser.add_argument("--subreddit", required=True, help="Subreddit to collect from")
    reddit_parser.add_argument("--limit", type=int, default=100, help="Number of users to collect")
    reddit_parser.add_argument("--output", default="./data/reddit", help="Output directory")

    # GitHub collection
    github_parser = collect_subparsers.add_parser("github", help="Collect from GitHub")
    github_parser.add_argument("--query", required=True, help="Search query for repositories")
    github_parser.add_argument("--limit", type=int, default=100, help="Number of users to collect")
    github_parser.add_argument("--output", default="./data/github", help="Output directory")

    # Wikipedia collection
    wiki_parser = collect_subparsers.add_parser("wikipedia", help="Collect from Wikipedia")
    wiki_parser.add_argument("--limit", type=int, default=100, help="Number of editors to collect")
    wiki_parser.add_argument("--output", default="./data/wikipedia", help="Output directory")

    args = parser.parse_args()

    if args.command == "check-env":
        check_env()
    elif args.command == "server":
        run_server(args.host, args.port, args.reload)
    elif args.command == "init-db":
        asyncio.run(init_database())
    elif args.command == "collect":
        if args.platform == "reddit":
            asyncio.run(collect_reddit_data(args.subreddit, args.limit, args.output))
        elif args.platform == "github":
            asyncio.run(collect_github_data(args.query, args.limit, args.output))
        elif args.platform == "wikipedia":
            asyncio.run(collect_wikipedia_data(args.limit, args.output))
        else:
            collect_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
