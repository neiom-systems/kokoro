#!/usr/bin/env python3
"""
Test script to verify Luxembourgish language support in Kokoro.

This script tests:
1. Import and initialization of Luxembourgish pipeline
2. G2P conversion of Luxembourgish text
3. Phonemization output format

Usage:
    python test_luxembourgish_integration.py
"""

import sys
from pathlib import Path

# Add kokoro to path if needed
sys.path.insert(0, str(Path(__file__).parent))


def test_luxembourgish_import():
    """Test that Luxembourgish can be imported and initialized."""
    print("=" * 80)
    print("Test 1: Import and Initialize Luxembourgish Pipeline")
    print("=" * 80)
    
    try:
        from kokoro import KPipeline
        
        # Test initialization with Luxembourgish
        print("\nInitializing KPipeline with lang_code='l'...")
        pipeline = KPipeline(lang_code='l', model=False)
        
        print("✓ Luxembourgish pipeline initialized successfully")
        print(f"✓ Language code: {pipeline.lang_code}")
        print(f"✓ G2P type: {type(pipeline.g2p).__name__}")
        
        return pipeline
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("\nInstall the custom Misaki fork:")
        print("  pip install git+https://github.com/neiom-systems/misaki.git")
        return None
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_g2p_conversion(pipeline):
    """Test G2P conversion of Luxembourgish text."""
    print("\n" + "=" * 80)
    print("Test 2: G2P Conversion")
    print("=" * 80)
    
    if pipeline is None:
        print("✗ Cannot test - pipeline is None")
        return
    
    test_texts = [
        "Moien",
        "Moien alleguer, wéi geet et?",
        "Gëschter ass e schéinen dag gewiescht.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        try:
            phonemes, _ = pipeline.g2p(text)
            print(f"  Phonemes: {phonemes}")
            
            if phonemes:
                print("  ✓ G2P conversion successful")
            else:
                print("  ⚠ Phonemes are empty")
        except Exception as e:
            print(f"  ✗ G2P conversion failed: {e}")


def test_phonemizer_compatibility():
    """Test that the custom Misaki LBG2P is compatible."""
    print("\n" + "=" * 80)
    print("Test 3: Direct Misaki LBG2P Usage")
    print("=" * 80)
    
    try:
        from misaki import lb
        
        g2p = lb.LBG2P()
        print("✓ Imported lb.LBG2P successfully")
        
        # Test with sample text
        text = "Moien allegenuer"
        phonemes, tokens = g2p(text)
        
        print(f"  Text: {text}")
        print(f"  Phonemes: {phonemes}")
        print(f"  Tokens: {tokens}")
        
        # Get statistics
        stats = g2p.get_stats()
        print(f"  Dictionary entries: {stats['total_entries']}")
        print(f"  Unknown symbol: {stats['unk_symbol']}")
        
    except ImportError as e:
        print(f"✗ Failed to import lb: {e}")
        print("\nInstall the custom Misaki fork:")
        print("  pip install git+https://github.com/neiom-systems/misaki.git")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_chunking(pipeline):
    """Test text chunking for long Luxembourgish text."""
    print("\n" + "=" * 80)
    print("Test 4: Text Chunking")
    print("=" * 80)
    
    if pipeline is None:
        print("✗ Cannot test - pipeline is None")
        return
    
    # Create a long text
    long_text = " ".join([
        "Mir treffen eis an enger stonn.",
        "Gëschter ass e schéinen dag gewiescht.",
        "D'Schoul ass ëmmer laanscht op.",
        "Ech ginn haut op de Maart.",
        "Wéi geet et dir?",
    ] * 3)
    
    print(f"Long text length: {len(long_text)} characters")
    
    # Test with pipeline (using the __call__ method)
    print("\nProcessing through pipeline (lang_code='l', model=False)...")
    
    try:
        results = list(pipeline(long_text))
        print(f"✓ Generated {len(results)} chunks")
        
        for i, result in enumerate(results, 1):
            print(f"  Chunk {i}:")
            print(f"    Graphemes: {result.graphemes[:50]}...")
            print(f"    Phonemes length: {len(result.phonemes)}")
            
            if len(result.phonemes) > 510:
                print(f"    ⚠ Phonemes exceed 510 chars: {len(result.phonemes)}")
            else:
                print(f"    ✓ Within 510 char limit")
    except Exception as e:
        print(f"✗ Chunking failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Luxembourgish Integration Tests")
    print("=" * 80)
    print()
    
    # Test 1: Import and initialize
    pipeline = test_luxembourgish_import()
    
    if pipeline:
        # Test 2: G2P conversion
        test_g2p_conversion(pipeline)
        
        # Test 3: Direct Misaki usage
        test_phonemizer_compatibility()
        
        # Test 4: Chunking
        test_chunking(pipeline)
    
    print("\n" + "=" * 80)
    print("Tests Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

