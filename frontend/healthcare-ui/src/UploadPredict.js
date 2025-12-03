import React, { useState } from "react";

export default function UploadPredict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult(data);
  }

  return (
    <div style={{ padding: "20px" }}>
      <h2>Lung Disease Detection</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={e => setFile(e.target.files[0])} />
        <button type="submit">Upload & Detect</button>
      </form>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>Probabilities:</strong> {JSON.stringify(result.probabilities)}</p>
        </div>
      )}
    </div>
  );
}
