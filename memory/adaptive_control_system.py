"""
AdaptiveControlSystem (adaptive_control_system.py)

Uses OpenAI GPT-4 for live classification of any text into hot/warm/cold.
"""

import os
import logging
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

class AdaptiveControlSystem:
    """
    Orchestrates memory-tier decisions using OpenAI GPT-4 mini
    """

    def __init__(
        self,
        hot_threshold_ms: float = 100.0,
        warm_threshold_ms: float = 500.0,
        cold_threshold_ms: float = 2000.0,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        
        # retrieval SLA targets in milliseconds
        self.hot_threshold_ms = hot_threshold_ms
        self.warm_threshold_ms = warm_threshold_ms
        self.cold_threshold_ms = cold_threshold_ms
        
        logger.info(
            f"AdaptiveControlSystem initialized with thresholds: "
            f"hot={self.hot_threshold_ms}ms, "
            f"warm={self.warm_threshold_ms}ms, "
            f"cold={self.cold_threshold_ms}ms"
        )

    def classify(self, text: str) -> str:
        """
        Uses GPT-4 mini to classify text into 'hot', 'warm', or 'cold' tiers
        """
        system_prompt = (
            "You are a smart router that assigns pieces of conversation "
            "to one of three memory tiers based on importance and recency:\n"
            "- hot: needs immediate access (current conversation context)\n"
            "- warm: recent but secondary context (summaries, key points)\n"
            "- cold: archived, rarely used context (old conversations)\n"
            "Respond with exactly one word: hot, warm, or cold."
        )
        
        user_prompt = f'Classify the following text:\n"""{text}"""'

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1,
            )
            
            tier = response.choices[0].message.content.strip().lower()
            
            if tier not in {"hot", "warm", "cold"}:
                logger.warning(f"Unexpected tier '{tier}', defaulting to 'warm'")
                return "warm"
                
            logger.info(f"Classified text as '{tier}'")
            return tier
            
        except Exception as e:
            logger.error(f"Classification error: {e}, defaulting to 'warm'")
            return "warm"

    def assign_tier(self, item: dict) -> str:
        """
        Takes a dict with at least a 'text' field and returns the assigned memory tier
        """
        text = item.get("text", "")
        return self.classify(text)

    def adjust_thresholds(self, usage_stats: dict[str, list[float]]) -> None:
        """
        Adjust thresholds based on observed retrieval latencies
        """
        for tier in ("hot", "warm", "cold"):
            times = usage_stats.get(tier, [])
            if not times:
                continue
                
            avg = sum(times) / len(times)
            new_threshold = avg * 1.2
            setattr(self, f"{tier}_threshold_ms", new_threshold)
            
            logger.info(
                f"Adjusted {tier} threshold to {new_threshold:.1f}ms "
                f"(avg observed {avg:.1f}ms)"
            )