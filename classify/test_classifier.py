"""
Test module for the classification functionality.

This module tests the article classification using sample cases with
predefined articles and summaries.
"""
import os
import pytest

from .classifier import classify_article, classify_article_simple, ClassificationResult
from common.llm_client import Llama, Mistral

# Global LLM instance - easily switch between models
# Change this to Mistral() to use Mistral instead of Llama
llm = Llama()


def test_llm_connection():
    """Test that the local LLM server is reachable and responding."""
    try:
        response = llm.simple("Hello, respond with just 'OK'", max_tokens=5)
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"LLM response: {response}")
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")


@pytest.mark.integration
def test_classify_fake_article():
    """Test classification of a clearly fake article."""
    try:
        # Test LLM connection
        llm.simple("Test", max_tokens=3)
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")
    
    # Sample fake article
    article_title = "Scientists Discover That Vaccines Cause Autism"
    article_content = (
        "A groundbreaking study published today reveals that vaccines are directly "
        "responsible for causing autism in children. The research, conducted by "
        "independent scientists, found that mercury in vaccines damages brain cells "
        "and leads to autism spectrum disorders. Parents are urged to avoid all "
        "vaccinations to protect their children from this devastating condition."
    )
    
    # Fake news evidence summary
    fake_summary = (
        "The evidence suggests that vaccines contain harmful substances like mercury "
        "and aluminum that can cause neurological damage. Multiple studies have shown "
        "correlations between vaccination schedules and autism rates. The medical "
        "establishment has been covering up these findings to protect pharmaceutical profits."
    )
    
    # Reliable news evidence summary
    reliable_summary = (
        "Extensive scientific research has consistently shown that vaccines do not "
        "cause autism. The original study linking vaccines to autism has been "
        "retracted and debunked. Major health organizations worldwide confirm that "
        "vaccines are safe and effective, with no credible evidence linking them to autism."
    )
    
    # Classify the article
    result = classify_article(
        llm=llm,
        article_title=article_title,
        article_content=article_content,
        fake_summary=fake_summary,
        reliable_summary=reliable_summary,
        temperature=0.1,
        max_tokens=300
    )
    
    # Assertions
    assert isinstance(result, ClassificationResult)
    assert result.prediction == "fake"
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 10
    assert isinstance(result.raw_response, str)
    
    # For this clearly fake article, we expect it to be classified as fake
    # (though we don't enforce this strictly as LLM responses can vary)
    print(f"Classification: {result.prediction}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")


@pytest.mark.integration
def test_classify_reliable_article():
    """Test classification of a clearly reliable article."""
    try:
        # Test LLM connection
        llm.simple("Test", max_tokens=3)
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")
    
    # Sample reliable article
    article_title = "New Study Confirms Climate Change is Human-Caused"
    article_content = (
        "A comprehensive study published in Nature Climate Change has confirmed "
        "that human activities are the primary driver of global warming. The research, "
        "conducted by an international team of scientists, analyzed temperature data "
        "from over 100 weather stations worldwide. The findings show that greenhouse "
        "gas emissions from human activities have caused the Earth's temperature to "
        "rise by approximately 1.1°C since pre-industrial times."
    )
    
    # Fake news evidence summary
    fake_summary = (
        "Climate change is a hoax created by global elites to control populations. "
        "Temperature data has been manipulated to show warming trends. Weather stations "
        "have been moved to urban heat islands, and historical data has been adjusted "
        "to fit the global warming narrative."
    )
    
    # Reliable news evidence summary
    reliable_summary = (
        "NASA and the IPCC have confirmed that human influence has warmed the climate "
        "at an unprecedented rate. The scientific consensus is based on thousands of "
        "peer-reviewed studies. Greenhouse gas emissions from human activities are "
        "the primary cause of observed warming trends."
    )
    
    # Classify the article
    result = classify_article(
        llm=llm,
        article_title=article_title,
        article_content=article_content,
        fake_summary=fake_summary,
        reliable_summary=reliable_summary,
        temperature=0.1,
        max_tokens=300
    )
    
    # Assertions
    assert isinstance(result, ClassificationResult)
    assert result.prediction == "reliable"
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 10
    assert isinstance(result.raw_response, str)
    
    # For this clearly reliable article, we expect it to be classified as reliable
    print(f"Classification: {result.prediction}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")


@pytest.mark.integration
def test_classify_ambiguous_article():
    """Test classification of an ambiguous article."""
    try:
        # Test LLM connection
        llm.simple("Test", max_tokens=3)
    except Exception as e:
        pytest.skip(f"Local LLM not reachable: {e}")
    
    # Sample ambiguous article
    article_title = "New Technology Promises to Revolutionize Energy Storage"
    article_content = (
        "A startup company claims to have developed a revolutionary battery technology "
        "that can store energy for weeks without degradation. The technology, which "
        "uses a novel approach to energy storage, could potentially solve the "
        "intermittency problem of renewable energy sources. However, the company "
        "has not yet published peer-reviewed research or demonstrated the technology "
        "at scale."
    )
    
    # Fake news evidence summary
    fake_summary = (
        "Many energy storage claims are exaggerated or fraudulent. Companies often "
        "make bold claims about breakthrough technologies to attract investment, "
        "but these technologies rarely work as advertised. The energy storage "
        "industry is full of scams and overhyped solutions."
    )
    
    # Reliable news evidence summary
    reliable_summary = (
        "Energy storage technology is advancing rapidly with legitimate innovations. "
        "While many claims need verification, the field has seen genuine breakthroughs. "
        "Proper due diligence and peer review are essential to distinguish real "
        "innovations from exaggerated claims."
    )
    
    # Classify the article
    result = classify_article(
        llm=llm,
        article_title=article_title,
        article_content=article_content,
        fake_summary=fake_summary,
        reliable_summary=reliable_summary,
        temperature=0.1,
        max_tokens=300
    )
    
    # Assertions
    assert isinstance(result, ClassificationResult)
    assert result.prediction in ["fake", "reliable"]
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 10
    
    print(f"Classification: {result.prediction}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")


def test_parse_classification_response():
    """Test the response parsing functionality."""
    from .classifier import _parse_classification_response
    
    # Test normal response
    response = """Classification: FAKE
Confidence: 0.85
Reasoning: The article contains multiple false claims that contradict reliable evidence."""
    
    result = _parse_classification_response(response)
    assert result.prediction == "fake"
    assert result.confidence == 0.85
    assert "false claims" in result.reasoning
    
    # Test reliable response
    response = """Classification: RELIABLE
Confidence: 0.9
Reasoning: The article aligns well with established scientific evidence."""
    
    result = _parse_classification_response(response)
    assert result.prediction == "reliable"
    assert result.confidence == 0.9
    assert "scientific evidence" in result.reasoning
    
    # Test malformed response
    response = "This is a malformed response without proper format."
    result = _parse_classification_response(response)
    assert result.prediction == "reliable"  # default
    assert result.confidence == 0.5  # default
    assert result.reasoning == "Unable to parse response"


if __name__ == "__main__":
    # Run a simple test if called directly
    print("Testing LLM connection...")
    test_llm_connection()
    print("Testing classification...")
    test_classify_fake_article()
    test_parse_classification_response()
    print("All tests passed!")
