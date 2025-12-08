import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [disease, setDisease] = useState("breast");

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
      // Change ' to ` (backtick)
const res = await axios.post(`https://healthcare-backend-lg75.onrender.com/predict/${organType}`, 
  formData, 
  { headers: { "Content-Type": "multipart/form-data" } }
);
      setResult(res.data.prediction);
    } catch (err) {
      console.error(err);
      alert("Error while processing image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-bg">
      <div className="glass-card">
        <h1 className="app-title">🩺 Multi-Organ Disease Detection</h1>
        <p className="app-subtitle">Upload your scan for instant analysis</p>

        <div className="dropdown-section">
          <label>Select Organ Type:</label>
          <select
            value={disease}
            onChange={(e) => setDisease(e.target.value)}
            className="dropdown"
          >
            <option value="lung">Lung</option>
            <option value="liver">Liver</option>
            <option value="breast">Breast</option>
          </select>
        </div>

        <div className="upload-box">
          <input
            type="file"
            accept="image/*"
            id="file-upload"
            onChange={handleFileChange}
          />
          <label htmlFor="file-upload" className="upload-btn">
            {file ? "📸 " + file.name : "Upload Image"}
          </label>
        </div>

        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Preview" />
          </div>
        )}

        <button
          onClick={handleUpload}
          className={`predict-btn ${loading ? "loading" : ""}`}
          disabled={loading}
        >
          {loading ? "🔍 Analyzing..." : "🚀 Predict"}
        </button>

        {result && (
          <div
            className={`result-card ${
              result.toLowerCase().includes("normal") ||
              result.toLowerCase().includes("benign")
                ? "result-normal"
                : "result-abnormal"
            }`}
          >
            <h2>Prediction Result:</h2>
            <p>{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
