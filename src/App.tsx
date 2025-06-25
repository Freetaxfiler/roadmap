import React from 'react';
import Header from './components/Header';
import PhaseCard from './components/PhaseCard';
import ResourcesSection from './components/ResourcesSection';
import { phases } from './data/roadmapData';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      {/* Phases Section */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">16-Week Learning Path</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Follow this structured approach to master data engineering skills progressively
            </p>
          </div>
          
          <div className="space-y-6">
            {phases.map((phase, index) => (
              <PhaseCard key={index} phase={phase} />
            ))}
          </div>
        </div>
      </section>

      <ResourcesSection />

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h3 className="text-2xl font-bold mb-4">Your Data Engineering Journey Starts Here</h3>
          <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
            Remember: consistency is key. Dedicate time daily, practice hands-on coding, and build projects. 
            The data engineering field rewards those who can bridge theory with practical implementation.
          </p>
          <div className="flex flex-col sm:flex-row justify-center items-center gap-4 text-sm text-gray-400">
            <span>Built for aspiring data engineers</span>
            <span className="hidden sm:block">•</span>
            <span>Follow the path, build the skills</span>
            <span className="hidden sm:block">•</span>
            <span>Join the data revolution</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;