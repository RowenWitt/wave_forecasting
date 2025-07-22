def test_chunked_approach():
    """Test the new chunked approach safely"""
    
    print("🧪 Testing Chunked Approach (Safe)")
    print("=" * 40)
    
    print("1. Existing approach still works:")
    print("   from experiments.train_production_spatial import run_production_spatial_experiment")
    print("   # Your current slow training continues unchanged")
    
    print("\n2. New chunked approach available:")
    print("   from experiments.train_chunked_spatial import run_chunked_spatial_experiment")
    print("   # New fast chunked training")
    
    print("\n3. Run new approach in parallel:")
    try:
        from experiments.train_chunked_spatial import run_chunked_spatial_experiment
        chunked_id = run_chunked_spatial_experiment()
        print(f"✅ Chunked training successful: {chunked_id}")
        
        print("\n4. If successful, can replace slow training")
        
    except Exception as e:
        print(f"❌ Chunked training failed: {e}")
        print("   Existing training continues unaffected")

if __name__ == "__main__":
    test_chunked_approach()