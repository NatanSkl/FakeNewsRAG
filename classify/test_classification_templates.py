"""
Test script to display all classification template combinations.

This script shows all combinations of:
- naming_convention: "fake_reliable", "type1_type2"
- prompt_type: 0, 1, 2
"""

import random
from classify.classifier import _generate_classification_template, CLASSIFICATION_SYSTEMS


def test_all_combinations():
    """Display all combinations of naming conventions and prompt types."""
    
    naming_conventions = ["fake_reliable", "type1_type2"]
    prompt_types = [0, 1, 2]
    
    # Sample article data for testing
    article_title = "Artificial Intelligence Breakthrough in Medical Diagnosis"
    article_content = "Researchers at Stanford University have developed a new AI system that can diagnose diseases with 95% accuracy. The system uses machine learning algorithms to analyze medical images and patient data."
    
    fake_summary = """A study found that doctors diagnose illnesses twice as accurately as online symptom-checking apps. 
New age medicine is being driven by common sense technologies. 
Medical mistakes contribute to 210,000-440,000 patient deaths per year in the US."""
    
    reliable_summary = """Dr. Dhaliwal's diagnostic method is similar to IBM's Deep Blue Chess project.
Myanmar health authorities struggle to prepare for Zika outbreak.
Computer-assisted diagnostics have a long history, with programs like Quick Medical Reference developed in the 1970s."""
    
    print("=" * 100)
    print("CLASSIFICATION TEMPLATE COMBINATIONS TEST")
    print("=" * 100)
    print()
    
    combination_num = 1
    for naming_convention in naming_conventions:
        for prompt_type in prompt_types:
            # Set random seed for reproducibility within each combination
            random.seed(42 + combination_num)
            
            print(f"\n{'=' * 100}")
            print(f"COMBINATION {combination_num}: naming_convention='{naming_convention}', prompt_type={prompt_type}")
            print(f"{'=' * 100}\n")
            
            # Show the system prompt for this prompt_type
            print(f"SYSTEM PROMPT (prompt_type={prompt_type}):")
            print("-" * 100)
            print(CLASSIFICATION_SYSTEMS[prompt_type].strip())
            print()
            
            # Generate the template
            template = _generate_classification_template(
                naming_convention=naming_convention,
                promt_type=prompt_type
            )
            
            # Fill in the template with sample data
            filled_template = template.format(
                article_title=article_title,
                article_content=article_content,
                fake_summary=fake_summary,
                reliable_summary=reliable_summary
            )
            
            print("USER PROMPT:")
            print("-" * 100)
            print(filled_template)
            print()
            
            combination_num += 1
    
    print("\n" + "=" * 100)
    print(f"Total combinations displayed: {len(naming_conventions) * len(prompt_types)}")
    print("=" * 100)


def test_randomization():
    """Test that randomization works (order and type1/type2 assignment)."""
    
    print("\n" + "=" * 100)
    print("RANDOMIZATION TEST")
    print("=" * 100)
    print("\nTesting 5 random generations for each naming convention to show randomization:\n")
    
    for naming_convention in ["fake_reliable", "type1_type2"]:
        print(f"\n{naming_convention.upper()}:")
        print("-" * 50)
        
        for i in range(5):
            # Don't set seed to see actual randomization
            template = _generate_classification_template(
                naming_convention=naming_convention,
                promt_type=0
            )
            
            # Check which appears first by looking for the keywords
            lines = template.split('\n')
            first_summary = None
            for line in lines:
                if 'FAKE' in line and 'NEWS' in line:
                    first_summary = 'FAKE'
                    break
                elif 'RELIABLE' in line and 'NEWS' in line:
                    first_summary = 'RELIABLE'
                    break
                elif 'TYPE1' in line:
                    first_summary = 'TYPE1'
                    break
                elif 'TYPE2' in line:
                    first_summary = 'TYPE2'
                    break
            
            print(f"  Generation {i+1}: First summary = {first_summary}")


if __name__ == "__main__":
    test_all_combinations()
    test_randomization()

