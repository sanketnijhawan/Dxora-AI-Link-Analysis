"""Typed request and response contracts for the link-analysis endpoint."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


PLATFORMS = {"Google", "Facebook", "Instagram", "LinkedIn", "X"}


class AnalysisRequest(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def website_only(cls, url: HttpUrl) -> HttpUrl:
        if url.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https")
        return url


class Brand(BaseModel):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    products_services: list[str] = Field(default_factory=list)


class Competitor(BaseModel):
    name: str = Field(min_length=1)
    website: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses_or_gaps: list[str] = Field(default_factory=list)
    source_basis: Literal["suggested_unverified", "verified"] = "suggested_unverified"
    evidence_urls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_required_for_verified(self):
        if self.source_basis == "verified" and not self.evidence_urls:
            raise ValueError("Verified competitors must include evidence URLs")
        return self


class CompetitorAnalysis(BaseModel):
    brand_overview: str
    competitors: list[Competitor]
    strengths: list[str]
    opportunities: list[str]
    research_note: str


class SearchQuery(BaseModel):
    query: str = Field(min_length=2)
    platforms: list[Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]] = Field(min_length=1)
    intent: str = Field(min_length=1)
    business_relevance: str
    target_audience: str


class SEOImprovements(BaseModel):
    primary_keyword: str
    secondary_keywords: list[str]
    long_tail_keywords: list[str]
    relevant_business_keywords: list[str]
    recommended_title: str
    meta_description: str
    supporting_content: list[str]


class Service(BaseModel):
    name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source_basis: Literal["observed", "suggested"]


class TargetAudience(BaseModel):
    primary_audience: list[str]
    secondary_audience: list[str]
    industries: list[str]
    business_types: list[str]
    customer_profiles: list[str]
    geographical_focus: str | None = None
    customer_needs: list[str]
    potential_interests: list[str]
    recommended_marketing_segments: list[str]


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: list[Service]
    target_audience: TargetAudience

    @model_validator(mode="after")
    def required_counts(self):
        if len(self.target_search_queries) != 5:
            raise ValueError("Exactly five target search queries are required")
        used = {p for query in self.target_search_queries for p in query.platforms}
        if used != PLATFORMS:
            raise ValueError("Five queries must collectively cover Google, Facebook, Instagram, LinkedIn and X")
        if len(self.top_services) != 5:
            raise ValueError("Exactly five services/recommendations are required")
        return self


class ReportBody(BaseModel):
    brand: Brand
    analysis: Analysis


class SiteEvidence(BaseModel):
    page_title: str | None
    meta_description: str | None
    headings: list[str]
    word_count: int
    pages_analyzed: int = 1


class ReportMetadata(BaseModel):
    model: str
    generated_at: datetime
    evidence: SiteEvidence
    notes: list[str]


class AnalysisResponse(BaseModel):
    success: Literal[True] = True
    website: str
    brand: Brand
    analysis: Analysis
    metadata: ReportMetadata
