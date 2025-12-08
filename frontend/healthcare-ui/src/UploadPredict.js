import React, { useState } from "react";

export default function UploadPredict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    // Assuming you have a state variable called 'selectedOrgan' or 'organType'
// (Use the variable name that stores the value from your dropdown)

const res = await fetch(`https://healthcare-backend-lg75.onrender.com/predict/${organType}`, {
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
