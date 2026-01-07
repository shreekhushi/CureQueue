import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [organType, setOrganType] = useState("breast"); // ✅ FIX

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult("");
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    setResult("");

    try {
      const res = await axios.post(
        `http://127.0.0.1:8000/predict/${organType}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(res.data.prediction);
    } catch (err) {
      console.error(err);
      alert("Error while processing image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <h1>🩺 Multi-Organ Disease Detection</h1>

        <label>Select Organ Type:</label>
        <select
          value={organType}
          onChange={(e) => setOrganType(e.target.value)}
        >
          <option value="breast">Breast</option>
          <option value="lung">Lung</option>
          <option value="liver">Liver</option>
        </select>

        <input type="file" accept="image/*" onChange={handleFileChange} />

        {preview && <img src={preview} alt="preview" className="preview-img" />}

        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Analyzing..." : "Predict"}
        </button>

        {result && (
          <div className="result-box">
            <h2>Prediction:</h2>
            <p>{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
