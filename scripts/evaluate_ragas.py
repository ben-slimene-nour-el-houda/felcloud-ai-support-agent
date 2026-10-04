import sys
import unittest.mock as mock
sys.modules['langchain_community.chat_models.vertexai'] = mock.MagicMock()

import json
import logging

from datasets import Dataset
from ragas.metrics.collections import answer_relevancy, faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.graph.workflow import app as langgraph_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluations():
    logger.info("Loading golden dataset...")
    try:
        with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error("Golden dataset not found.")
        return

    # Only evaluate RAG metrics on 'support' intents that fetch context
    support_queries = [item for item in dataset if item.get("expected_intent") == "support"]
    
    if not support_queries:
        logger.warning("No support queries found for RAG evaluation.")
        return

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "language": []
    }

    logger.info(f"Generating answers for {len(support_queries)} support queries...")
    for item in support_queries:
        question = item["input"]
        language = item.get("language", "en")
        
        state_input = {
            "conversation_id": f"eval_{language}",
            "session_id": f"eval_{language}",
            "channel": "webhook",
            "user_message": question,
            "language": language,
            "chat_history": []
        }
        
        try:
            result = langgraph_app.invoke(state_input)
            
            answer = result.get("final_response") or ""
            
            sources = result.get("sources")
            if sources:
                contexts = [src.get("content", str(src)) for src in sources]
            else:
                retrieved = result.get("retrieved_context") or ""
                contexts = [retrieved] if retrieved else ["No context retrieved"]

            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts)
            eval_data["language"].append(language)
            
            logger.info(f"[{language.upper()}] Processed: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to process query '{question[:30]}...': {e}")
            continue

    if not eval_data["question"]:
        logger.warning("No valid data generated for evaluation.")
        return

    hf_dataset = Dataset.from_dict(eval_data)
    
    evaluator_llm = ChatOpenAI(
        model=settings.STRUCTURED_OUTPUT_MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        temperature=0.0
    )
    
    evaluator_embeddings = OpenAIEmbeddings(
        model="bge-m3",
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        check_embedding_ctx_length=False
    )
    
    logger.info("Running RAGAS evaluation...")
    try:
        result = evaluate(
            dataset=hf_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        logger.info("=== Evaluation Complete! ===")
        logger.info(f"RAGAS Results: {result}")
        
        # Build a plain serialisable dict
        scores = {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"]
        }
        output = {"overall": scores, "per_language": {}}
        
        # Per-language breakdown
        try:
            df = result.to_pandas()
            df["language"] = eval_data["language"][:len(df)]
            for lang in df["language"].unique():
                lang_df = df[df["language"] == lang]
                output["per_language"][lang] = {
                    "faithfulness": round(float(lang_df["faithfulness"].mean()), 4),
                    "answer_relevancy": round(float(lang_df["answer_relevancy"].mean()), 4),
                    "num_samples": len(lang_df)
                }
                logger.info(
                    f"  [{lang.upper()}] faithfulness={output['per_language'][lang]['faithfulness']}, "
                    f"answer_relevancy={output['per_language'][lang]['answer_relevancy']}"
                )
        except Exception as per_lang_err:
            logger.warning(f"Per-language breakdown failed: {per_lang_err}")
        
        # Save results
        with open("data/ragas_evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2)
            
        logger.info("Results saved to data/ragas_evaluation_results.json")
        
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluations()
