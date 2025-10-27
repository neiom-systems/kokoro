#!/usr/bin/env python3
"""
Comprehensive analysis of Kokoro-82M model structure and voice files.
This script details the structure of voice embeddings and model weights.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


class KokoroStructureAnalyzer:
    """Analyzer for Kokoro model and voice file structure."""
    
    def __init__(self, base_model_path: str = "base_model"):
        self.base_path = Path(base_model_path)
        self.voices_path = self.base_path / "voices"
        self.model_path = self.base_path / "kokoro-v1_0.pth"
        self.config_path = self.base_path / "config.json"
    
    def analyze_voice_files(self, sample_voice: str = "af_heart.pt") -> Dict:
        """
        Analyze voice file structure in detail.
        
        Args:
            sample_voice: Voice file to analyze
            
        Returns:
            Dictionary with detailed voice file information
        """
        print("\n" + "="*80)
        print("VOICE FILES ANALYSIS")
        print("="*80)
        
        voice_file = self.voices_path / sample_voice
        
        if not voice_file.exists():
            print(f"❌ Voice file not found: {voice_file}")
            return {}
        
        # Load voice embedding
        voice_tensor = torch.load(voice_file, weights_only=True)
        
        analysis = {
            "file_name": sample_voice,
            "file_path": str(voice_file),
            "file_size_mb": voice_file.stat().st_size / (1024 * 1024),
        }
        
        print(f"\n📁 Voice File: {sample_voice}")
        print(f"   File Path: {voice_file}")
        print(f"   File Size: {analysis['file_size_mb']:.2f} MB")
        
        # Tensor properties
        print(f"\n📊 Tensor Properties:")
        print(f"   Shape: {voice_tensor.shape}")
        print(f"   Data Type: {voice_tensor.dtype}")
        print(f"   Device: {voice_tensor.device}")
        print(f"   Total Elements: {voice_tensor.numel():,}")
        
        analysis.update({
            "tensor_shape": tuple(voice_tensor.shape),
            "tensor_dtype": str(voice_tensor.dtype),
            "total_elements": voice_tensor.numel(),
        })
        
        # Statistical analysis
        voice_numpy = voice_tensor.cpu().numpy() if voice_tensor.is_cuda else voice_tensor.numpy()
        
        stats = {
            "min": float(voice_numpy.min()),
            "max": float(voice_numpy.max()),
            "mean": float(voice_numpy.mean()),
            "std": float(voice_numpy.std()),
            "median": float(np.median(voice_numpy)),
        }
        
        print(f"\n📈 Statistical Analysis:")
        print(f"   Min Value: {stats['min']:.6f}")
        print(f"   Max Value: {stats['max']:.6f}")
        print(f"   Mean: {stats['mean']:.6f}")
        print(f"   Std Dev: {stats['std']:.6f}")
        print(f"   Median: {stats['median']:.6f}")
        
        analysis["statistics"] = stats
        
        # Dimension interpretation
        if len(voice_tensor.shape) == 3:
            seq_len, batch_size, embedding_dim = voice_tensor.shape
            print(f"\n📐 Dimension Interpretation (3D Tensor):")
            print(f"   Sequence Length: {seq_len} (max phoneme positions)")
            print(f"   Batch Size: {batch_size}")
            print(f"   Embedding Dimension: {embedding_dim}")
            print(f"   → Purpose: Pre-computed speaker embeddings for each timestep")
            analysis["dimensions"] = {
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "embedding_dimension": embedding_dim,
            }
        elif len(voice_tensor.shape) == 2:
            dim1, dim2 = voice_tensor.shape
            print(f"\n📐 Dimension Interpretation (2D Tensor):")
            print(f"   Dimension 1: {dim1}")
            print(f"   Dimension 2: {dim2}")
            analysis["dimensions"] = {
                "dim1": dim1,
                "dim2": dim2,
            }
        
        # Distribution visualization
        print(f"\n📊 Value Distribution:")
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        for q in quantiles:
            value = float(np.quantile(voice_numpy, q))
            print(f"   {q*100:5.1f}% percentile: {value:8.4f}")
        
        return analysis
    
    def analyze_all_voices(self) -> Dict:
        """Analyze all voice files for consistency."""
        print("\n" + "="*80)
        print("ANALYZING ALL VOICE FILES")
        print("="*80)
        
        voice_files = sorted(list(self.voices_path.glob("*.pt")))
        print(f"\n📂 Found {len(voice_files)} voice files")
        
        all_voices_info = {
            "total_voices": len(voice_files),
            "voices": {}
        }
        
        # Analyze shape consistency
        shapes = {}
        for voice_file in voice_files:
            voice_tensor = torch.load(voice_file, weights_only=True)
            shape = tuple(voice_tensor.shape)
            if shape not in shapes:
                shapes[shape] = []
            shapes[shape].append(voice_file.name)
        
        print(f"\n🔍 Voice File Shapes:")
        for shape, files in shapes.items():
            print(f"   Shape {shape}: {len(files)} voices")
            if len(files) <= 5:
                for f in files:
                    print(f"      - {f}")
            else:
                for f in files[:3]:
                    print(f"      - {f}")
                print(f"      ... and {len(files)-3} more")
        
        all_voices_info["shape_distribution"] = {str(k): len(v) for k, v in shapes.items()}
        
        return all_voices_info
    
    def analyze_model_weights(self) -> Dict:
        """
        Analyze model weights structure in detail.
        
        Returns:
            Dictionary with detailed model weight information
        """
        print("\n" + "="*80)
        print("MODEL WEIGHTS ANALYSIS")
        print("="*80)
        
        if not self.model_path.exists():
            print(f"❌ Model file not found: {self.model_path}")
            return {}
        
        print(f"\n📁 Loading Model: {self.model_path.name}")
        print(f"   File Size: {self.model_path.stat().st_size / (1024**3):.2f} GB")
        
        # Load model weights
        model_weights = torch.load(self.model_path, weights_only=True, map_location='cpu')
        
        print(f"\n✅ Model loaded successfully")
        print(f"   Total Top-level Keys: {len(model_weights)}")
        
        # Flatten and organize weights by component
        components, flat_weights = self._organize_weights(model_weights)
        
        print(f"\n📦 Model Components:")
        total_params = 0
        for component, info in components.items():
            if info['num_keys'] == 0:
                continue
            print(f"\n   {component.upper()}")
            print(f"      Keys: {info['num_keys']}")
            print(f"      Parameters: {info['total_params']:,}")
            total_params += info['total_params']
            
            # Show sample keys
            if info['keys'][:3]:
                print(f"      Sample Keys:")
                for key in info['keys'][:3]:
                    tensor = flat_weights[key]
                    if hasattr(tensor, 'shape'):
                        shape = tensor.shape
                        dtype = tensor.dtype
                        print(f"         - {key}: shape={shape}, dtype={dtype}")
                if len(info['keys']) > 3:
                    print(f"         ... and {len(info['keys'])-3} more keys")
        
        print(f"\n📊 Total Parameters: {total_params:,}")
        
        return {
            "total_keys": len(model_weights),
            "components": components,
            "total_parameters": total_params,
        }
    
    def _organize_weights(self, weights: Dict) -> Tuple[Dict, Dict]:
        """Organize weights by component and return flattened weights."""
        components = {
            "bert": {"keys": [], "total_params": 0, "num_keys": 0},
            "bert_encoder": {"keys": [], "total_params": 0, "num_keys": 0},
            "text_encoder": {"keys": [], "total_params": 0, "num_keys": 0},
            "decoder": {"keys": [], "total_params": 0, "num_keys": 0},
            "predictor": {"keys": [], "total_params": 0, "num_keys": 0},
            "other": {"keys": [], "total_params": 0, "num_keys": 0},
        }
        
        flat_weights = {}
        
        # Recursively flatten nested dictionaries
        def flatten_weights(d, prefix=""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_weights(value, full_key)
                else:
                    flat_weights[full_key] = value
        
        flatten_weights(weights)
        
        for key, tensor in flat_weights.items():
            # Handle both tensors and non-tensor values
            if hasattr(tensor, 'numel'):
                params = tensor.numel()
            elif isinstance(tensor, torch.Tensor):
                params = tensor.numel()
            else:
                continue
            
            # Classify by prefix
            if "bert" in key and "bert_encoder" not in key:
                components["bert"]["keys"].append(key)
                components["bert"]["total_params"] += params
                components["bert"]["num_keys"] += 1
            elif "bert_encoder" in key:
                components["bert_encoder"]["keys"].append(key)
                components["bert_encoder"]["total_params"] += params
                components["bert_encoder"]["num_keys"] += 1
            elif "text_encoder" in key:
                components["text_encoder"]["keys"].append(key)
                components["text_encoder"]["total_params"] += params
                components["text_encoder"]["num_keys"] += 1
            elif "decoder" in key:
                components["decoder"]["keys"].append(key)
                components["decoder"]["total_params"] += params
                components["decoder"]["num_keys"] += 1
            elif "predictor" in key:
                components["predictor"]["keys"].append(key)
                components["predictor"]["total_params"] += params
                components["predictor"]["num_keys"] += 1
            else:
                components["other"]["keys"].append(key)
                components["other"]["total_params"] += params
                components["other"]["num_keys"] += 1
        
        return components, flat_weights
    
    def analyze_config(self) -> Dict:
        """Analyze model configuration."""
        print("\n" + "="*80)
        print("MODEL CONFIGURATION ANALYSIS")
        print("="*80)
        
        if not self.config_path.exists():
            print(f"❌ Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n📋 Configuration Keys: {len(config)}")
        
        # Pretty print config
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"\n   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            else:
                print(f"\n   {key}: {value}")
        
        return config
    
    def run_full_analysis(self):
        """Run complete analysis."""
        print("\n🔬 KOKORO-82M MODEL STRUCTURE ANALYSIS")
        print("="*80)
        
        voice_analysis = self.analyze_voice_files()
        all_voices = self.analyze_all_voices()
        model_analysis = self.analyze_model_weights()
        config_analysis = self.analyze_config()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            "voice_sample": voice_analysis,
            "all_voices": all_voices,
            "model_weights": model_analysis,
            "config": config_analysis,
        }


def main():
    """Main execution."""
    analyzer = KokoroStructureAnalyzer()
    results = analyzer.run_full_analysis()
    
    # Save results to markdown file
    output_file = Path("specs/STRUCTURE_ANALYSIS_RESULTS.md")
    
    with open(output_file, 'w') as f:
        f.write("# Kokoro-82M Model Structure Analysis Results\n\n")
        f.write(f"Generated: {Path('base_model/kokoro-v1_0.pth').stat().st_ctime}\n\n")
        
        # Voice Sample Analysis
        if results.get("voice_sample"):
            f.write("## Voice File Analysis (Sample: af_heart.pt)\n\n")
            voice = results["voice_sample"]
            f.write(f"- **File Path**: {voice.get('file_path')}\n")
            f.write(f"- **File Size**: {voice.get('file_size_mb'):.2f} MB\n")
            f.write(f"- **Tensor Shape**: {voice.get('tensor_shape')}\n")
            f.write(f"- **Data Type**: {voice.get('tensor_dtype')}\n")
            f.write(f"- **Total Elements**: {voice.get('total_elements'):,}\n\n")
            
            if voice.get("statistics"):
                f.write("### Statistical Properties\n\n")
                stats = voice["statistics"]
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Min | {stats['min']:.6f} |\n")
                f.write(f"| Max | {stats['max']:.6f} |\n")
                f.write(f"| Mean | {stats['mean']:.6f} |\n")
                f.write(f"| Std Dev | {stats['std']:.6f} |\n")
                f.write(f"| Median | {stats['median']:.6f} |\n\n")
            
            if voice.get("dimensions"):
                f.write("### Dimension Interpretation\n\n")
                dims = voice["dimensions"]
                f.write(f"- **Sequence Length**: {dims.get('sequence_length')} (max phoneme positions)\n")
                f.write(f"- **Batch Size**: {dims.get('batch_size')}\n")
                f.write(f"- **Embedding Dimension**: {dims.get('embedding_dimension')}\n")
                f.write(f"- **Purpose**: Pre-computed speaker embeddings for each timestep\n\n")
        
        # All Voices Analysis
        if results.get("all_voices"):
            f.write("## All Voice Files Analysis\n\n")
            all_voices = results["all_voices"]
            f.write(f"- **Total Voices**: {all_voices.get('total_voices')}\n\n")
            f.write("### Shape Distribution\n\n")
            for shape, count in all_voices.get('shape_distribution', {}).items():
                f.write(f"- Shape `{shape}`: {count} voices\n")
            f.write("\n")
        
        # Model Weights Analysis
        if results.get("model_weights"):
            f.write("## Model Weights Analysis\n\n")
            weights = results["model_weights"]
            f.write(f"- **Total Parameters**: {weights.get('total_parameters'):,}\n")
            f.write(f"- **Total Top-level Keys**: {weights.get('total_keys')}\n\n")
            
            f.write("### Component Breakdown\n\n")
            f.write("| Component | Keys | Parameters |\n")
            f.write("|-----------|------|------------|\n")
            
            components = weights.get("components", {})
            for comp_name, comp_info in components.items():
                if comp_info['num_keys'] > 0:
                    f.write(f"| {comp_name.upper()} | {comp_info['num_keys']} | {comp_info['total_params']:,} |\n")
            f.write("\n")
            
            # Detailed component info
            f.write("### Detailed Component Structure\n\n")
            for comp_name, comp_info in components.items():
                if comp_info['num_keys'] == 0:
                    continue
                
                f.write(f"#### {comp_name.upper()}\n\n")
                f.write(f"- **Number of Keys**: {comp_info['num_keys']}\n")
                f.write(f"- **Total Parameters**: {comp_info['total_params']:,}\n")
                f.write(f"- **Sample Keys**:\n\n")
                f.write("```\n")
                for key in comp_info['keys'][:5]:
                    f.write(f"  {key}\n")
                if len(comp_info['keys']) > 5:
                    f.write(f"  ... and {len(comp_info['keys']) - 5} more keys\n")
                f.write("```\n\n")
        
        # Configuration Analysis
        if results.get("config"):
            f.write("## Model Configuration\n\n")
            config = results["config"]
            
            for key, value in config.items():
                if isinstance(value, dict):
                    f.write(f"### {key.upper()}\n\n")
                    f.write("```json\n")
                    import json as json_module
                    f.write(json_module.dumps(value, indent=2))
                    f.write("\n```\n\n")
                else:
                    f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("### Voice Embeddings\n")
        f.write("- **Shape**: [510, 1, 256]\n")
        f.write("  - 510 sequence positions (max phoneme tokens)\n")
        f.write("  - 1 batch size\n")
        f.write("  - 256-dimensional speaker embeddings\n")
        f.write("- **Purpose**: Pre-computed speaker characteristics at each timestep for conditioning the model\n")
        f.write("- **Value Range**: [-1.51, 1.76] with mean ≈ 0\n\n")
        
        f.write("### Model Architecture (81.7M parameters)\n")
        f.write("1. **BERT** (6.3M params) - Linguistic encoder with 12 layers, 768 hidden dim\n")
        f.write("2. **BERT Encoder** (394K params) - Maps BERT output to model hidden dimension (512)\n")
        f.write("3. **Text Encoder** (11.5M params) - Additional text processing with LSTMs\n")
        f.write("4. **Decoder** (53.3M params) - ISTFTNet vocoder for audio reconstruction\n")
        f.write("5. **Predictor** (10.3M params) - Duration, F0, and prosody prediction\n\n")
        
        f.write("### Key Specifications\n")
        f.write("- **Max Sequence Length**: 512 tokens\n")
        f.write("- **Mel Spectrogram Dimension**: 80\n")
        f.write("- **Phoneme Vocabulary**: 178 tokens\n")
        f.write("- **Speaker Embedding Size**: 128 (+ 256-dim vectors in voice files)\n")
        f.write("- **Multispeaker Support**: Yes\n")
        f.write("- **Supported Languages**: English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese\n\n")
    
    print(f"\n✨ Full results saved to: {output_file}")


if __name__ == "__main__":
    main()
