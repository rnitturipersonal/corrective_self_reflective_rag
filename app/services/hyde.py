from app.services.llm_service import LLMService
from app.config import get_settings
from loguru import logger
import json


class HydeService:
    """
    Hypothetical Document Embeddings (HYDE) service.

    Generates multiple hypothetical answer documents for a query to improve retrieval
    by bridging the semantic gap between short queries and document content.
    """

    def __init__(self):
        self.settings = get_settings()
        self.llm = LLMService()

    def generate_hypothetical_documents(
        self,
        query: str,
        num_hypotheses: int = None
    ) -> list[str]:
        """
        Generate hypothetical answer documents using LLM.

        Args:
            query: User's query
            num_hypotheses: Number of hypotheses to generate (default from settings)

        Returns:
            List of hypothetical document passages
            Falls back to [query] if generation fails
        """
        if num_hypotheses is None:
            num_hypotheses = self.settings.hyde_num_hypotheses

        system_prompt = """You are an expert at generating hypothetical document passages that could answer a given question.
Generate diverse, detailed passages that represent different ways the question could be answered in actual documents.
Each passage should be 2-3 sentences long and read like an excerpt from a real document.
Return your response as a JSON object with a single key "hypotheses" containing an array of passages."""

        user_prompt = f"""Generate {num_hypotheses} different hypothetical document passages that could answer this question:

Question: {query}

Each passage should:
1. Be 2-3 sentences long
2. Provide a different perspective or aspect of the answer
3. Read like an excerpt from an actual document
4. Be informative and specific

Return ONLY a JSON object in this format:
{{
  "hypotheses": [
    "First hypothetical passage...",
    "Second hypothetical passage...",
    "Third hypothetical passage..."
  ]
}}"""

        try:
            response = self.llm.generate_with_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7  # Higher temperature for diversity
            )

            # Parse JSON response
            result = json.loads(response)
            hypotheses = result.get("hypotheses", [])

            # Validate we got hypotheses
            if not hypotheses or not isinstance(hypotheses, list):
                logger.warning("HYDE: Invalid hypotheses format, falling back to query")
                return [query]

            # Filter out empty strings
            hypotheses = [h for h in hypotheses if h and isinstance(h, str)]

            if not hypotheses:
                logger.warning("HYDE: No valid hypotheses generated, falling back to query")
                return [query]

            logger.info(f"HYDE: Generated {len(hypotheses)} hypothetical documents")
            return hypotheses

        except json.JSONDecodeError as e:
            logger.error(f"HYDE: JSON parsing error: {e}, falling back to query")
            return [query]
        except Exception as e:
            logger.error(f"HYDE: Hypothesis generation error: {e}, falling back to query")
            return [query]
