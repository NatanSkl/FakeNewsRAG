#!/usr/bin/env python3
"""
Simple test for query_once function.

This test verifies that query_once can load metadata and perform searches
on a FAISS index.
"""

import argparse
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from retrieval_v3 import query_once, embed_and_normalize, load_store


def load_real_store(store_dir="/StudentData/index"):
    """Load the real store from the specified directory."""
    print(f"Loading real store from {store_dir}...")
    
    # Load the store using load_store function
    store = load_store(store_dir, verbose=True)
    
    print(f"Loaded store with {store.index.ntotal} vectors")
    print(f"Index type: {type(store.index)}")
    print(f"Index dimension: {store.index.d}")
    print(f"BM25 available: {store.bm25 is not None}")
    print(f"v2d mappings: {len(store.v2d)}")
    print(f"Original data shape: {store.original.shape}")
    
    return store


def test_query_once_basic():
    """Test basic query_once functionality."""
    print("\n" + "="*60)
    print("TEST: Basic query_once functionality")
    print("="*60)
    
    # Load real store
    store = load_real_store("/StudentData/index")
    
    # Test queries
    test_queries = [
        "artificial intelligence machine learning",
        "healthcare technology innovation",
        "fake news misinformation",
        "christian education foundation",
        """
        The peace plan proposes an immediate end to fighting and the release within 72 hours of 20 living Israeli hostages held by Hamas - as well as the remains of hostages thought to be dead - in exchange for hundreds of detained Gazans.

In a statement, Hamas said it also "renews its agreement to hand over the administration of the Gaza Strip to a Palestinian body of independents (technocrats), based on Palestinian national consensus and Arab and Islamic support."

But it added that the part of the proposals dealing with the future of Gaza and the rights of Palestinian people was still being discussed "within a national framework".

Earlier on Friday, Trump posted on his Truth Social platform: "If this LAST CHANCE agreement is not reached, all HELL, like no one has ever seen before, will break out against Hamas. THERE WILL BE PEACE IN THE MIDDLE EAST ONE WAY OR THE OTHER," Trump wrote in the Truth Social post.

On Tuesday Trump had said that he was giving Hamas "three to four days" to respond to the peace plan.

There are believed to be 48 hostages still being held in the Palestinian territory by the armed group, only 20 of whom are thought to be alive.

In a briefing at the White House on Friday afternoon, Press Secretary Karoline Leavitt said that the consequences of turning down the deal would be "very grave" for Hamas.

"I think that the entire world should hear the president of the United States loud and clear," Leavitt added. "Hamas has an opportunity to accept this plan and move forward in a peaceful and prosperous manner in the region. If they don't, the consequences, unfortunately, are going to be very tragic."

The 20-point plan, agreed by Trump and Israeli Prime Minister Benjamin Netanyahu and announced by both at the White House on Monday, also says Hamas will have no role in governing Gaza, and leaves the door open for an eventual Palestinian state.

However, Netanyahu later reinstated his longstanding opposition to a Palestinian state, saying in a video statement shortly after the announcement: "It's not written in the agreement. We said we would strongly oppose a Palestinian state."

The plan stipulates that once both sides agree to the proposal "full aid will be immediately sent into the Gaza Strip".

It also outlines a plan for the future governance of Gaza, saying a "technocratic, apolitical Palestinian committee" will govern temporarily "with oversight and supervision by a new international transitional body, called the Board of Peace", which it says will be headed by Trump.

    Read the full plan 

European and Middle Eastern leaders have welcomed the proposal. The Palestinian Authority (PA), which governs parts of the Israeli-occupied West Bank, has called the US president's efforts "sincere and determined".

Pakistan initially voiced support for the plan, but the country's foreign minister has since said the points announced were not in line with a draft from a group of Muslim-majority countries, BBC Urdu and Reuters reported.

Trump has said that if Hamas does not agree to the plan, Israel would have US backing to "finish the job of destroying the threat of Hamas".

Netanyahu has also said Israel "will finish the job" if Hamas rejected the plan or did not follow through.

The Israeli military launched a campaign in Gaza in response to the Hamas-led attack on southern Israel on 7 October 2023, in which about 1,200 people were killed and 251 others were taken hostage.

At least 66,288 people have been killed in Israeli attacks in Gaza since then, according to the territory's Hamas-run health ministry.

In the 24 hours before Friday midday, 63 people were killed by Israeli military operations, the health ministry said.

The push for the peace plan comes as Israel is carrying out an offensive in Gaza City, with Israel's defence minister saying earlier this week that Israeli forces were "tightening the siege" around the city.

Israel has said the offensive aims to secure the release of the remaining hostages.

Hundreds of thousands of Gaza City residents have been forced to flee after the Israeli military ordered evacuations to a designated "humanitarian area" in the southern al-Mawasi area, but hundreds of thousands more are believed to have remained.

Israel's defence minister has warned that those who stay during the offensive against Hamas would be "terrorists and supporters of terror".

James Elder, spokesman for the UN children's agency, Unicef, said on Friday that the idea of a safe zone in southern Gaza was "farcical".

"Bombs are dropped from the sky with chilling predictability. Schools, which have been designated as temporary shelters, are regularly reduced to rubble," he said.
        """
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        try:
            results = query_once(store, query, k=5)
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  [{i}] Score: {result['score']:.4f}")
                print(f"      Text: {result['content'][:80]}...")
                print(f"      Label: {result['label']}")
                print(f"      DB ID: {result['db_id']}")
                print(f"      Title: {result.get('title', 'N/A')[:50]}...")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()




def main():
    """Run the basic test with real index."""
    parser = argparse.ArgumentParser(description="Test query_once function with real index")
    
    print("Starting query_once test with real index...")
    print("="*60)
    
    try:
        test_query_once_basic()
        
        print("\n" + "="*60)
        print("🎉 query_once test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
