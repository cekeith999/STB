import urllib.request
import urllib.error
import json
import tempfile
import pathlib
import shutil
import bpy

def test_meshy_api():
    # Replace this with your actual API key
    API_KEY = "msy_YOUR_API_KEY_HERE"  # Replace with your actual key
    
    base_url = "https://api.meshy.ai"
    endpoint = "/openapi/v2/text-to-3d"
    
    # Test payload according to Meshy docs
    payload = {
        "mode": "preview",
        "prompt": "simple low-poly test object",
        "ai_model": "meshy-5",
        "should_remesh": True
    }
    
    print(f"Testing Meshy API...")
    print(f"URL: {base_url}{endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make the request
        url = f"{base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        body = json.dumps(payload).encode("utf-8")
        
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        
        print(f"\nSending request...")
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(f"✅ SUCCESS! Response: {json.dumps(result, indent=2)}")
            
            # Extract job ID
            job_id = result.get("result")
            if job_id:
                print(f"\nJob ID: {job_id}")
                print(f"Now checking status...")
                
                # Check status
                status_url = f"{base_url}/openapi/v2/text-to-3d/{job_id}"
                status_req = urllib.request.Request(status_url, method="GET", headers=headers)
                
                with urllib.request.urlopen(status_req, timeout=30) as status_response:
                    status_result = json.loads(status_response.read().decode("utf-8"))
                    print(f"Status: {json.dumps(status_result, indent=2)}")
                    
                    # If succeeded, try to download and import
                    if status_result.get("status") == "SUCCEEDED":
                        print(f"\nJob succeeded! Downloading model...")
                        
                        model_urls = status_result.get("model_urls", {})
                        if "glb" in model_urls:
                            download_url = model_urls["glb"]
                            print(f"Downloading GLB from: {download_url}")
                            
                            # Download to temp file
                            tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="meshy_test_"))
                            local_path = tmpdir / "test_model.glb"
                            
                            with urllib.request.urlopen(download_url, timeout=300) as download_response:
                                with open(local_path, "wb") as f:
                                    shutil.copyfileobj(download_response, f)
                            
                            print(f"Downloaded to: {local_path}")
                            
                            # Import into Blender
                            print(f"Importing into Blender...")
                            before_objects = set(bpy.data.objects)
                            
                            try:
                                bpy.ops.import_scene.gltf(filepath=str(local_path))
                                after_objects = set(bpy.data.objects)
                                new_objects = after_objects - before_objects
                                
                                print(f"✅ Successfully imported {len(new_objects)} objects!")
                                for obj in new_objects:
                                    print(f"  - {obj.name}")
                                    
                            except Exception as import_error:
                                print(f"❌ Import failed: {import_error}")
                                try:
                                    bpy.ops.wm.gltf_import(filepath=str(local_path))
                                    after_objects = set(bpy.data.objects)
                                    new_objects = after_objects - before_objects
                                    print(f"✅ Successfully imported {len(new_objects)} objects (alternative method)!")
                                except Exception as import_error2:
                                    print(f"❌ Alternative import also failed: {import_error2}")
                        else:
                            print(f"No GLB URL found in model_urls: {model_urls}")
                    else:
                        print(f"Job not yet succeeded. Status: {status_result.get('status')}")
                        
            return True
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"❌ HTTP Error {e.code}: {error_body}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Run the test
if __name__ == "__main__":
    print("=== Meshy API Direct Test ===")
    print("Make sure to replace 'msy_YOUR_API_KEY_HERE' with your actual API key!")
    print()
    
    # Check if API key was replaced
    if "YOUR_API_KEY_HERE" in test_meshy_api.__code__.co_consts[0]:
        print("⚠️  WARNING: You need to replace the API key in the script first!")
        print("   Edit the API_KEY variable with your actual Meshy API key.")
    else:
        test_meshy_api()
