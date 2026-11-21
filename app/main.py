"""FastAPI entry point. Run: uvicorn app.main:app --host 0.0.0.0 --port 4000"""
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from app.schemas import AnalysisRequest, AnalysisResponse, ReportMetadata, SiteEvidence
from app.services.llm_service import LLMServiceError, generate_analysis
from app.services.website_extractor import WebsiteFetchError, fetch_website

app = FastAPI(title="AI-Powered Website Link Analysis", version="1.0.0", description="One-page website intelligence with structured LLM insights.")


@app.post("/link-analysis", response_model=AnalysisResponse)
async def link_analysis(payload: AnalysisRequest) -> AnalysisResponse:
    try:
        site = await fetch_website(str(payload.url))
        body, model = await generate_analysis(site)
    except (WebsiteFetchError, LLMServiceError) as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    return AnalysisResponse(
        website=site.url,
        brand=body.brand,
        analysis=body.analysis,
        metadata=ReportMetadata(
            model=model,
            generated_at=datetime.now(timezone.utc),
            evidence=SiteEvidence(
                page_title=site.title,
                meta_description=site.meta_description,
                headings=site.headings,
                word_count=site.word_count,
                pages_analyzed=1,
            ),
            notes=[
                "Only the submitted page was fetched; this is not a full-site audit.",
                "Competitor suggestions are unverified; manually research before using business claims.",
                "Search queries are relevance suggestions, not measured search-volume data.",
                "Services labeled suggested are not verified current offerings.",
            ],
        ),
    )
