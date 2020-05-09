import io
import os
import sys
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import time
import logging

import basnet

logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)


# Simple probe.
@app.route('/', methods=['GET'])
def hello():
    return """
<!doctype html>
<head>
    <title>BASNet</title>
</head>
<body>
    <h1>Upload image</h1>
    <form method="POST" enctype="multipart/form-data">
        <p><input type="file" name="data"></p>
        <p><input type="checkbox" id="mask" name="mask" value="true"><label for="mask"> Mask only</label></p>
        <p><input type="submit" value="Get image"></p>
    </form>
</body>
</html>
"""


# Route http posts to this method
@app.route('/', methods=['POST'])
def run():
    start = time.time()

    # Convert string of image data to uint8
    if 'data' not in request.files:
        return jsonify({'error': 'missing file param `data`'}), 400
    data = request.files['data'].read()
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    return_mask_only = request.form.get('mask', 'false') == 'true'

    # Convert string data to PIL Image
    img = Image.open(io.BytesIO(data))
    original_size = img.size

    # Ensure image size is under 1024
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.thumbnail((1024, 1024))

    # Process Image
    res = basnet.run(np.array(img))
    mask = res.resize(original_size).convert("L")
    output_img = mask

    # Apply mask if needed
    if not return_mask_only:
        ref = Image.open(io.BytesIO(data))
        empty = Image.new("RGBA", ref.size, 0)
        applied_mask = Image.composite(ref, empty, mask)
        output_img = applied_mask

    # Save to buffer
    buff = io.BytesIO()
    output_img.save(buff, 'PNG')
    buff.seek(0)

    # Print stats
    logging.info(f'Completed in {time.time() - start:.2f}s')

    # Return data
    return send_file(
        buff,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='output.png'
    )


if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'production'
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
