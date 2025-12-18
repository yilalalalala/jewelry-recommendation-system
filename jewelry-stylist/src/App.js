import React, { useState, useRef } from 'react';
import { Upload, Sparkles, ChevronRight, X } from 'lucide-react';

const JewelryStyleApp = () => {
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
  const fileInputRef = useRef(null);

  const styles = [
    'classic', 'modern', 'romantic', 'bold',
    'bohemian', 'minimalist', 'vintage', 'luxurious'
  ];

  const materials = ['Gold', 'Silver', 'Platinum', 'Rose Gold'];

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

  const toggleStyle = (style) => {
    setPreferences(prev => ({
      ...prev,
      styles: prev.styles.includes(style)
        ? prev.styles.filter(s => s !== style)
        : [...prev.styles, style]
    }));
  };

  const handleGetRecommendations = async () => {
    setLoading(true);
    
    try {
      // Call Supabase Edge Function
      const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
      const supabaseKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
      
      const response = await fetch(`${supabaseUrl}/functions/v1/get-recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${supabaseKey}`,
          'apikey': supabaseKey
        },
        body: JSON.stringify({
          image: uploadedImage,
          preferences: preferences
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('API Response:', data);
      
      setRecommendations(data.recommendations || []);
      setStep(3);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      alert('Failed to get recommendations. Please try again.');
      
      // Fallback to mock data for testing
      setRecommendations([]);
      setStep(2);
    }
    
    setLoading(false);
  };

  const resetApp = () => {
    setStep(1);
    setUploadedImage(null);
    setPreferences({
      styles: [],
      budgetMin: 100,
      budgetMax: 5000,
      material: '',
      event: ''
    });
    setRecommendations([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Parisienne&display=swap');
      `}</style>
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl tracking-wide" style={{ fontFamily: 'Parisienne, cursive', color: '#D4AF37' }}>
              Maison de Bijoux
            </h1>
            {step > 1 && (
              <button
                onClick={resetApp}
                className="text-slate-400 hover:text-slate-200 transition-colors text-sm tracking-wider"
              >
                START OVER
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Step 1: Upload Image */}
        {step === 1 && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-light text-slate-100 mb-4 tracking-wide">
                Your Personal <span style={{ fontFamily: 'Parisienne, cursive', color: '#D4AF37' }}>Cartier</span> Jewelry Stylist
              </h2>
              <p className="text-slate-400 text-lg">
                Upload your outfit and discover perfectly curated jewelry pieces
              </p>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              className="relative border-2 border-dashed border-slate-600 rounded-lg p-16 text-center cursor-pointer hover:border-slate-500 transition-all group bg-slate-800/30"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <Upload className="w-16 h-16 mx-auto mb-6 text-slate-500 group-hover:text-slate-400 transition-colors" />
              <p className="text-slate-300 text-lg mb-2">Upload Your Outfit</p>
              <p className="text-slate-500 text-sm">
                Click to select or drag and drop an image
              </p>
            </div>

            <div className="mt-12 flex items-center justify-center gap-2 text-slate-500 text-sm">
              <Sparkles className="w-4 h-4" />
              <span>Personal recommendations based on your style</span>
            </div>
          </div>
        )}

        {/* Step 2: Preferences */}
        {step === 2 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Left: Image Preview */}
            <div className="space-y-6">
              <h2 className="text-2xl font-light text-slate-100 tracking-wide">Your Outfit</h2>
              <div className="relative rounded-lg overflow-hidden bg-slate-800 aspect-[3/4]">
                <img
                  src={uploadedImage}
                  alt="Uploaded outfit"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>

            {/* Right: Preferences Form */}
            <div className="space-y-8">
              <h2 className="text-2xl font-light text-slate-100 tracking-wide">Style Preferences</h2>

              {/* Style Selection */}
              <div>
                <label className="block text-slate-300 mb-4 text-sm tracking-wider uppercase">
                  Select Styles
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {styles.map(style => (
                    <button
                      key={style}
                      onClick={() => toggleStyle(style)}
                      className={`px-4 py-3 rounded border text-sm tracking-wider uppercase transition-all ${
                        preferences.styles.includes(style)
                          ? 'bg-slate-100 text-slate-900 border-slate-100'
                          : 'bg-slate-800/50 text-slate-400 border-slate-700 hover:border-slate-600'
                      }`}
                    >
                      {style}
                    </button>
                  ))}
                </div>
              </div>

              {/* Budget Range */}
              <div>
                <label className="block text-slate-300 mb-4 text-sm tracking-wider uppercase">
                  Budget Range
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <input
                      type="number"
                      value={preferences.budgetMin}
                      onChange={(e) => setPreferences({...preferences, budgetMin: parseInt(e.target.value)})}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded text-slate-100 focus:border-slate-500 focus:outline-none"
                      placeholder="Min"
                    />
                  </div>
                  <div>
                    <input
                      type="number"
                      value={preferences.budgetMax}
                      onChange={(e) => setPreferences({...preferences, budgetMax: parseInt(e.target.value)})}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded text-slate-100 focus:border-slate-500 focus:outline-none"
                      placeholder="Max"
                    />
                  </div>
                </div>
              </div>

              {/* Material */}
              <div>
                <label className="block text-slate-300 mb-4 text-sm tracking-wider uppercase">
                  Preferred Material
                </label>
                <select
                  value={preferences.material}
                  onChange={(e) => setPreferences({...preferences, material: e.target.value})}
                  className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded text-slate-100 focus:border-slate-500 focus:outline-none"
                >
                  <option value="">Any Material</option>
                  {materials.map(mat => (
                    <option key={mat} value={mat}>{mat}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleGetRecommendations}
                disabled={preferences.styles.length === 0}
                className="w-full bg-slate-100 text-slate-900 py-4 rounded hover:bg-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-light tracking-wider uppercase flex items-center justify-center gap-2"
              >
                Get Recommendations
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Recommendations */}
        {step === 3 && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-3xl font-light text-slate-100 mb-2 tracking-wide">
                Your Curated Collection
              </h2>
              <p className="text-slate-400">
                {Math.min(recommendations.length, 6)} pieces selected exclusively for you
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {recommendations.slice(0, 6).map(item => (
                <div
                  key={item.id}
                  className="bg-slate-800/30 border border-slate-700/50 rounded-lg overflow-hidden hover:border-slate-600 transition-all group"
                >
                  <div className="aspect-[4/3] bg-slate-800 overflow-hidden">
                    <img
                      src={item.image_url}
                      alt={item.name}
                      className="w-full h-full object-contain p-4 group-hover:scale-105 transition-transform duration-500"
                    />
                  </div>
                  <div className="p-6 space-y-3">
                    <h3 className="text-lg font-light text-slate-100 tracking-wide min-h-[3rem]">
                      {item.name}
                    </h3>
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-light text-slate-100">
                        ${item.price.toLocaleString()}
                      </span>
                      <span className="text-sm text-slate-400">
                        {Math.min(item.match_score*10, 100)}% Match
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {item.style_tags?.slice(0, 3).map(tag => (
                        <span
                          key={tag}
                          className="px-2 py-1 bg-slate-700/50 text-slate-300 text-xs rounded uppercase tracking-wider"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <a 
                      href={`https://www.google.com/search?q=${encodeURIComponent(`Cartier ${item.name}`)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block w-full mt-4 py-3 bg-slate-700/50 hover:bg-slate-700 text-slate-100 rounded transition-colors text-sm tracking-wider uppercase text-center"
                    >
                      View Details
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-slate-100 mx-auto mb-4"></div>
              <p className="text-slate-100 tracking-wider">Curating your collection...</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default JewelryStyleApp;