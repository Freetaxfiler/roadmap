import React from 'react';
import { BookOpen, Clock, Target } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-700 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center">
          <div className="flex justify-center mb-6">
            <div className="bg-white/10 backdrop-blur-sm p-4 rounded-full">
              <BookOpen className="h-12 w-12" />
            </div>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
            Data Engineering Learning Roadmap
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-blue-100 max-w-3xl mx-auto">
            A comprehensive 16-week step-by-step guide to becoming a skilled data engineer
          </p>
          <div className="flex flex-col sm:flex-row justify-center items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              <span>16 Weeks Total</span>
            </div>
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              <span>4 Progressive Phases</span>
            </div>
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              <span>Production-Ready Skills</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;