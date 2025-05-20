import os
import tempfile
import urllib.request


def fetch_model(url: str, output_path: str = None) -> str:
	"""
	Download an ONNX model from `url` into `output_path` (if provided)
	or into a temporary file, and return the path to that file.
	"""
	# download bytes
	with urllib.request.urlopen(url) as resp:
		data = resp.read()

	if output_path:
		# ensure the directory exists
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		with open(output_path, "wb") as f:
			f.write(data)
		return output_path
	else:
		# write to temp file so onnx.load (or anything else) can read it
		with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
			tmp.write(data)
			tmp.flush()
			return tmp.name