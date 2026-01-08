// import React, { useState } from "react";

// export default function UploadPredict() {
//   const [file, setFile] = useState(null);
//   const [result, setResult] = useState(null);

//   async function handleSubmit(e) {
//     e.preventDefault();
//     if (!file) return;
//     const formData = new FormData();
//     formData.append("file", file);

//     // Assuming you have a state variable called 'selectedOrgan' or 'organType'
// // (Use the variable name that stores the value from your dropdown)

// // Make sure you are using backticks (`) not single quotes (')
// const res = await fetch(`https://detection-jvr6.onrender.com/predict/${organType}`, {
//     method: "POST",
//     body: formData,
// });
//     const data = await res.json();
//     setResult(data);
//   }

//   return (
//     <div style={{ padding: "20px" }}>
//       <h2>Lung Disease Detection</h2>
//       <form onSubmit={handleSubmit}>
//         <input type="file" onChange={e => setFile(e.target.files[0])} />
//         <button type="submit">Upload & Detect</button>
//       </form>

//       {result && (
//         <div style={{ marginTop: "20px" }}>
//           <p><strong>Prediction:</strong> {result.prediction}</p>
//           <p><strong>Probabilities:</strong> {JSON.stringify(result.probabilities)}</p>
//         </div>
//       )}
//     </div>
//   );
// }



import React, { useState } from "react";

export default function UploadPredict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [organType, setOrganType] = useState("lung");

  // Backend URL from Vercel environment variable
  const API_URL = process.env.REACT_APP_API_URL;

  async function handleSubmit(e) {
    e.preventDefault();

    if (!file) {
      alert("Please select an image file");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(
        `${API_URL}/predict/${organType}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) {
        throw new Error("Failed to process image");
      }

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Error while processing image");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "20px", maxWidth: "420px", margin: "auto" }}>
      <h2>Multi-Organ Disease Detection</h2>

      {/* Organ selection */}
      <label><strong>Select Organ Type:</strong></label>
      <select
        value={organType}
        onChange={(e) => setOrganType(e.target.value)}
        style={{ width: "100%", marginBottom: "12px" }}
      >
        <option value="lung">Lung</option>
        <option value="breast">Breast</option>
        <option value="liver">Liver</option>
      </select>

      {/* File upload */}
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ marginBottom: "10px" }}
        />

        <button type="submit" disabled={loading} style={{ width: "100%" }}>
          {loading ? "Analyzing..." : "Upload & Detect"}
        </button>
      </form>

      {/* Result display */}
      {result && (
        <div style={{ marginTop: "20px" }}>
          <p>
            <strong>Organ:</strong> {organType.toUpperCase()}
          </p>
          <p>
            <strong>Prediction:</strong> {result.prediction}
          </p>
          <p>
            <strong>Probabilities:</strong>{" "}
            {JSON.stringify(result.probabilities)}
          </p>
        </div>
      )}
    </div>
  );
}
