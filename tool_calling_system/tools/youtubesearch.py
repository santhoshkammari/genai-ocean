import json
from typing import Optional, List, Dict, Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools import BaseTool
from youtube_search import YoutubeSearch


class EnhancedYouTubeSearchTool(BaseTool):
    """Tool that queries YouTube and returns comprehensive video information."""

    name: str = "youtube_search"
    description: str = (
        "search for youtube videos associated with a person",
        "Useful for when you need video information"
    )

    def _search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        results = YoutubeSearch(query, num_results).to_dict()
        return results

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        values = query.split(",")
        search_query = values[0].strip()
        # num_results = int(values[1].strip()) if len(values) > 1 else 1
        num_results = 1

        search_results = self._search(search_query, num_results)

        formatted_results = []
        for video in search_results:
            formatted_video = {
                "title": video.get("title"),
                "url": f"https://www.youtube.com{video.get('url_suffix')}",
                # "description": video.get("long_desc"),
                # "channel": video.get("channel"),
                # "duration": video.get("duration"),
                # "views": video.get("views"),
                # "publish_time": video.get("publish_time"),
                # "thumbnail": video.get("thumbnails")[0] if video.get("thumbnails") else None
            }
            formatted_results.append(formatted_video)

        return json.dumps(formatted_results, indent=2)


youtube_search = EnhancedYouTubeSearchTool()

if __name__ == '__main__':
    user_input = "LISA - ROCKSTAR (Official Music Video)"
    print(youtube_search.run(user_input))