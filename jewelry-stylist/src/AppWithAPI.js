import React, { useState } from 'react';
import { Upload, Sparkles, ShoppingBag, ChevronRight, X, Loader2 } from 'lucide-react';
import './App.css';


const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Available style options (matches backend STYLE_PROFILES)
const STYLE_OPTIONS = [
  'classic', 'modern', 'romantic', 'bold', 
  'bohemian', 'minimalist', 'vintage', 'luxurious'
];

const MATERIAL_OPTIONS = ['Gold', 'Silver', 'Platinum', 'Rose Gold'];

function App() {
  const [step, setStep] = useState(1);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [preferences, setPreferences] = useState({
    styles: [],
    budgetMin: 100,
    budgetMax: 5000,
    material: ''
  });
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result);
        setStep(2);
      };
      reader.readAsDataURL(file);
    }
  };

  // Toggle style selection
  const toggleStyle = (style) => {
    setPreferences(prev => ({
      ...prev,
      styles: prev.styles.includes(style)
        ? prev.styles.filter(s => s !== style)
        : [...prev.styles, style]
    }));
  };

  // Get recommendations from FastAPI backend
  const handleGetRecommendations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/get-recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: uploadedImage,
          preferences: preferences
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setStep(3);
    } catch (err) {
      console.error('Failed to get recommendations:', err);
      setError('Failed to get recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Reset to start
  const handleReset = () => {
    setStep(1);
    setUploadedImage(null);
    setPreferences({
      styles: [],
      budgetMin: 100,
      budgetMax: 5000,
      material: ''
    });
    setRecommendations([]);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-8 h-8 text-purple-600" />
            <h1 className="text-2xl font-bold text-gray-800">Jewelry Stylist</h1>
          </div>
          <div className="text-sm text-gray-500">
            Powered by AI + SQLAlchemy ORM
          </div>
        </div>
      </header>

      {/* Progress Steps */}
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="flex items-center justify-center gap-4 mb-8">
          {[1, 2, 3].map((s) => (
            <div key={s} className="flex items-center">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold
                ${step >= s ? 'bg-purple-600 text-white' : 'bg-gray-200 text-gray-500'}`}>
                {s}
              </div>
              {s < 3 && <ChevronRight className="w-6 h-6 text-gray-400 mx-2" />}
            </div>
          ))}
        </div>

        {/* Step 1: Upload Image */}
        {step === 1 && (
          <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
            <Upload className="w-16 h-16 text-purple-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Your Outfit</h2>
            <p className="text-gray-600 mb-6">Share a photo of your outfit to get matching jewelry recommendations</p>
            
            <label className="inline-block cursor-pointer">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <span className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-3 rounded-lg font-semibold inline-flex items-center gap-2 transition-colors">
                <Upload className="w-5 h-5" />
                Choose Photo
              </span>
            </label>
            
            <p className="text-sm text-gray-500 mt-4">
              Or skip this step and go directly to preferences
            </p>
            <button
              onClick={() => setStep(2)}
              className="text-purple-600 hover:text-purple-700 font-medium mt-2"
            >
              Skip to preferences →
            </button>
          </div>
        )}

        {/* Step 2: Preferences */}
        {step === 2 && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Your Preferences</h2>
            
            {/* Uploaded Image Preview */}
            {uploadedImage && (
              <div className="mb-6 relative inline-block">
                <img src={uploadedImage} alt="Uploaded outfit" className="w-32 h-32 object-cover rounded-lg" />
                <button
                  onClick={() => setUploadedImage(null)}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Style Selection */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Select Styles (choose one or more)
              </label>
              <div className="flex flex-wrap gap-2">
                {STYLE_OPTIONS.map((style) => (
                  <button
                    key={style}
                    onClick={() => toggleStyle(style)}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-colors capitalize
                      ${preferences.styles.includes(style)
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                  >
                    {style}
                  </button>
                ))}
              </div>
            </div>

            {/* Budget Range */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Budget Range: ${preferences.budgetMin.toLocaleString()} - ${preferences.budgetMax.toLocaleString()}
              </label>
              <div className="flex gap-4 items-center">
                <input
                  type="range"
                  min="100"
                  max="50000"
                  step="100"
                  value={preferences.budgetMin}
                  onChange={(e) => setPreferences(prev => ({
                    ...prev,
                    budgetMin: Math.min(parseInt(e.target.value), prev.budgetMax - 100)
                  }))}
                  className="flex-1"
                />
                <input
                  type="range"
                  min="100"
                  max="50000"
                  step="100"
                  value={preferences.budgetMax}
                  onChange={(e) => setPreferences(prev => ({
                    ...prev,
                    budgetMax: Math.max(parseInt(e.target.value), prev.budgetMin + 100)
                  }))}
                  className="flex-1"
                />
              </div>
              <div className="flex justify-between text-sm text-gray-500 mt-1">
                <span>$100</span>
                <span>$50,000</span>
              </div>
            </div>

            {/* Material Preference */}
            <div className="mb-8">
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Preferred Material (optional)
              </label>
              <div className="flex flex-wrap gap-2">
                {MATERIAL_OPTIONS.map((material) => (
                  <button
                    key={material}
                    onClick={() => setPreferences(prev => ({
                      ...prev,
                      material: prev.material === material ? '' : material
                    }))}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-colors
                      ${preferences.material === material
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                  >
                    {material}
                  </button>
                ))}
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={() => setStep(1)}
                className="flex-1 px-6 py-3 border border-gray-300 rounded-lg font-semibold text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleGetRecommendations}
                disabled={loading || preferences.styles.length === 0}
                className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-300 text-white px-6 py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Finding matches...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Get Recommendations
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Recommendations */}
        {step === 3 && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Your Recommendations</h2>
              <button
                onClick={handleReset}
                className="text-purple-600 hover:text-purple-700 font-medium"
              >
                Start Over
              </button>
            </div>

            {recommendations.length === 0 ? (
              <div className="text-center py-12">
                <ShoppingBag className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No matching items found. Try adjusting your preferences.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recommendations.map((item) => (
                  <div key={item.id} className="border rounded-xl overflow-hidden hover:shadow-lg transition-shadow">
                    <div className="aspect-square bg-gray-100 relative">
                      {item.image_url ? (
                        <img
                          src={item.image_url}
                          alt={item.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <ShoppingBag className="w-12 h-12 text-gray-300" />
                        </div>
                      )}
                      <div className="absolute top-2 right-2 bg-purple-600 text-white text-xs font-bold px-2 py-1 rounded-full">
                        {Math.round(item.match_score * 100)}% match
                      </div>
                    </div>
                    <div className="p-4">
                      <h3 className="font-semibold text-gray-800 mb-1">{item.name}</h3>
                      <p className="text-lg font-bold text-purple-600 mb-2">
                        ${item.price.toLocaleString()}
                      </p>
                      <div className="flex flex-wrap gap-1 mb-3">
                        {item.style_tags?.map((tag) => (
                          <span
                            key={tag}
                            className="text-xs bg-purple-50 text-purple-700 px-2 py-1 rounded-full capitalize"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <a
                        href={`https://www.cartier.com/en-us/jewelry/all-jewelry/CR${item.ref}.html`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block text-center bg-gray-900 hover:bg-gray-800 text-white py-2 rounded-lg text-sm font-medium transition-colors"
                      >
                        View Details
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-auto py-6 text-center text-sm text-gray-500">
        <p>CSCI-GA.2433 Database Systems - Jewelry Recommendation Project</p>
        <p className="mt-1">Cao & Ngo • Part 4: End-to-End with ORM</p>
      </footer>
    </div>
  );
}

export default App;
