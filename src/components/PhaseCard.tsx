import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Calendar, Code, Database, Cloud } from 'lucide-react';

interface WeekData {
  title: string;
  days: string[];
  codeExample?: string;
  language?: string;
}

interface PhaseCardProps {
  phase: {
    number: number;
    title: string;
    weeks: string;
    description: string;
    weeks_data: WeekData[];
    color: string;
    icon: React.ReactNode;
  };
}

const PhaseCard: React.FC<PhaseCardProps> = ({ phase }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedWeek, setExpandedWeek] = useState<number | null>(null);

  const getColorClasses = (color: string) => {
    const colors = {
      blue: 'bg-blue-50 border-blue-200 text-blue-900',
      purple: 'bg-purple-50 border-purple-200 text-purple-900',
      green: 'bg-green-50 border-green-200 text-green-900',
      orange: 'bg-orange-50 border-orange-200 text-orange-900',
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  const getGradientClasses = (color: string) => {
    const gradients = {
      blue: 'from-blue-500 to-blue-600',
      purple: 'from-purple-500 to-purple-600', 
      green: 'from-green-500 to-green-600',
      orange: 'from-orange-500 to-orange-600',
    };
    return gradients[color as keyof typeof gradients] || gradients.blue;
  };

  return (
    <div className={`border-2 rounded-xl transition-all duration-300 hover:shadow-lg ${getColorClasses(phase.color)}`}>
      <div 
        className="p-6 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-lg bg-gradient-to-r ${getGradientClasses(phase.color)} text-white`}>
              {phase.icon}
            </div>
            <div>
              <h3 className="text-xl font-bold">Phase {phase.number}: {phase.title}</h3>
              <p className="text-sm opacity-70">{phase.weeks}</p>
            </div>
          </div>
          {isExpanded ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
        </div>
        <p className="mt-4 text-sm">{phase.description}</p>
      </div>

      {isExpanded && (
        <div className="border-t border-current border-opacity-20">
          <div className="p-6 space-y-4">
            {phase.weeks_data.map((week, weekIndex) => (
              <div key={weekIndex} className="bg-white/50 rounded-lg border border-current border-opacity-20">
                <div 
                  className="p-4 cursor-pointer flex items-center justify-between"
                  onClick={() => setExpandedWeek(expandedWeek === weekIndex ? null : weekIndex)}
                >
                  <h4 className="font-semibold flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    {week.title}
                  </h4>
                  {expandedWeek === weekIndex ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </div>
                
                {expandedWeek === weekIndex && (
                  <div className="px-4 pb-4 space-y-3">
                    <ul className="space-y-2">
                      {week.days.map((day, dayIndex) => (
                        <li key={dayIndex} className="text-sm flex items-start gap-2">
                          <div className="w-2 h-2 bg-current rounded-full mt-2 opacity-60 flex-shrink-0"></div>
                          <span>{day}</span>
                        </li>
                      ))}
                    </ul>
                    
                    {week.codeExample && (
                      <div className="mt-4">
                        <div className="bg-gray-900 rounded-lg p-4 text-sm font-mono text-gray-100 overflow-x-auto">
                          <div className="flex items-center gap-2 mb-2 text-gray-400">
                            <Code className="h-4 w-4" />
                            <span className="text-xs uppercase">{week.language || 'Code'}</span>
                          </div>
                          <pre className="whitespace-pre-wrap text-green-400">{week.codeExample}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PhaseCard;