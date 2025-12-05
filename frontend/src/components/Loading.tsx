import React, { useState } from 'react';

export default function CareerLoadingComponent() {
  const [isLoading, setIsLoading] = useState(true);

  const simulateLoading = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 4000);
  };

  if (!isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50">
        <div className="text-center">
          <div className="text-teal-500 mb-4 animate-bounce">
            <svg className="w-20 h-20 mx-auto" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Prediction Ready!</h2>
          <p className="text-gray-600 mb-6">Your career insights have been generated</p>
          <button
            onClick={simulateLoading}
            className="px-8 py-3 bg-teal-500 text-white rounded-full hover:bg-teal-600 transition font-semibold"
          >
            Analyze Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50">
      <div className="w-full max-w-2xl px-4">
        
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-12 text-center border border-teal-100">
       
          <div className="mb-8 flex justify-center">
            <div className="relative w-32 h-32">
              
              <div className="absolute inset-0 border-2 border-cyan-300 rounded-full"></div>
              
             
              <svg className="absolute inset-0 w-full h-full animate-spin" style={{ animationDuration: '2s' }} viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="url(#gradient)" strokeWidth="5" strokeDasharray="70 220" strokeLinecap="round" />
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#14b8a6" />
                    <stop offset="100%" stopColor="#06b6d4" />
                  </linearGradient>
                </defs>
              </svg>

              
              <div className="absolute inset-0 flex items-center justify-center">
                <svg className="w-16 h-16 animate-bounce" viewBox="0 0 100 100" style={{ animationDuration: '1.5s' }}>
                
                  <circle cx="50" cy="40" r="20" fill="#14b8a6" />
                  
                  
                  <circle cx="43" cy="37" r="2.5" fill="white" />
                  <circle cx="57" cy="37" r="2.5" fill="white" />
                  <circle cx="42" cy="36" r="1.2" fill="#0f766e" />
                  <circle cx="56" cy="36" r="1.2" fill="#0f766e" />
                  
                 
                  <path d="M 43 43 Q 50 46 57 43" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                  
                  
                  <rect x="40" y="60" width="20" height="18" rx="3" fill="#06b6d4" />
                  
                 
                  <rect x="28" y="65" width="10" height="6" rx="3" fill="#06b6d4" />
                  <rect x="62" y="65" width="10" height="6" rx="3" fill="#06b6d4" />
                  
                  
                  <rect x="43" y="78" width="5" height="10" rx="2" fill="#14b8a6" />
                  <rect x="52" y="78" width="5" height="10" rx="2" fill="#14b8a6" />
                </svg>
              </div>
              

            </div>
          </div>

         
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Predicting Your Career
          </h2>
          <p className="text-gray-600 text-lg mb-10">
            Let Logic AI analyze your profile and discover the perfect career paths for you...
          </p>

        
          <div className="mb-8">
            <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan-400 via-teal-500 to-green-500 rounded-full animate-pulse" style={{
                animation: 'loadingBar 2.5s ease-in-out infinite'
              }}></div>
            </div>
          </div>

        
          <div className="space-y-4 mb-10">
            <div className="flex items-center justify-start px-4">
              <div className="w-3 h-3 bg-teal-500 rounded-full mr-4 flex-shrink-0"></div>
              <div className="text-left flex-1">
                <span className="text-gray-700 font-medium">Analyzing your skills & experience</span>
              </div>
              <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            
            <div className="flex items-center justify-start px-4">
              <div className="w-3 h-3 bg-cyan-400 rounded-full mr-4 flex-shrink-0 animate-pulse"></div>
              <div className="text-left flex-1">
                <span className="text-gray-700 font-medium">Matching with career paths</span>
              </div>
              <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
            </div>
            
            <div className="flex items-center justify-start px-4">
              <div className="w-3 h-3 bg-gray-300 rounded-full mr-4 flex-shrink-0"></div>
              <div className="text-left flex-1">
                <span className="text-gray-500 font-medium">Generating personalized insights</span>
              </div>
              <svg className="w-5 h-5 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
              </svg>
            </div>
          </div>

        
          <div className="flex items-center justify-center space-x-1 mb-2">
            <span className="text-gray-600 font-medium">Thinking</span>
            <div className="flex space-x-1">
              <span className="w-2.5 h-2.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
              <span className="w-2.5 h-2.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></span>
              <span className="w-2.5 h-2.5 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></span>
            </div>
          </div>

          
          <p className="text-sm text-gray-500">Estimated time: 3-5 seconds</p>
        </div>

    
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/60 backdrop-blur rounded-2xl p-4 text-center border border-teal-100">
            <p className="text-sm text-gray-600"><span className="font-semibold text-teal-600"> Instant </span> Responses</p>
          </div>
          <div className="bg-white/60 backdrop-blur rounded-2xl p-4 text-center border border-teal-100">
            <p className="text-sm text-gray-600"><span className="font-semibold text-teal-600"> Private </span> & Secure</p>
          </div>
          <div className="bg-white/60 backdrop-blur rounded-2xl p-4 text-center border border-teal-100">
            <p className="text-sm text-gray-600"><span className="font-semibold text-teal-600"> Creative</span> & Logical</p>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes loadingBar {
          0% { width: 0%; }
          50% { width: 100%; }
          100% { width: 100%; }
        }
      `}</style>
    </div>
  );
}