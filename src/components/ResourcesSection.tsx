import React from 'react';
import { BookOpen, Video, Code, Award, Globe, Users } from 'lucide-react';

const ResourcesSection: React.FC = () => {
  const resourceCategories = [
    {
      title: "Learning Platforms",
      icon: <Video className="h-5 w-5" />,
      resources: [
        { name: "Udemy", description: "Comprehensive courses", color: "bg-red-500" },
        { name: "Coursera", description: "University-level content", color: "bg-blue-500" },
        { name: "Pluralsight", description: "Technical depth", color: "bg-pink-500" },
        { name: "YouTube", description: "Free tutorials", color: "bg-red-600" }
      ]
    },
    {
      title: "Practice Platforms",
      icon: <Code className="h-5 w-5" />,
      resources: [
        { name: "Kaggle", description: "Datasets and competitions", color: "bg-cyan-500" },
        { name: "HackerRank", description: "SQL and Python practice", color: "bg-green-500" },
        { name: "LeetCode", description: "Algorithm practice", color: "bg-orange-400" },
        { name: "AWS/Azure Free Tier", description: "Cloud practice", color: "bg-yellow-500" }
      ]
    },
    {
      title: "Essential Books",
      icon: <BookOpen className="h-5 w-5" />,
      resources: [
        { name: "Designing Data-Intensive Applications", description: "by Martin Kleppmann", color: "bg-purple-500" },
        { name: "Learning Spark", description: "by Holden Karau", color: "bg-indigo-500" },
        { name: "Python for Data Analysis", description: "by Wes McKinney", color: "bg-green-600" }
      ]
    }
  ];

  const schedule = {
    title: "Daily Schedule Recommendation",
    sessions: [
      { time: "Morning (2 hours)", activities: ["Theory and tutorials", "Documentation reading"] },
      { time: "Evening (1-2 hours)", activities: ["Hands-on practice", "Project work"] },
      { time: "Weekend (4-6 hours)", activities: ["Longer projects", "Review and consolidation"] }
    ]
  };

  return (
    <section className="py-16 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Learning Resources & Schedule</h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Curated resources and recommended schedule to maximize your learning efficiency
          </p>
        </div>

        {/* Resources Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
          {resourceCategories.map((category, index) => (
            <div key={index} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
                  {category.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900">{category.title}</h3>
              </div>
              <div className="space-y-4">
                {category.resources.map((resource, resourceIndex) => (
                  <div key={resourceIndex} className="flex items-start gap-3">
                    <div className={`w-3 h-3 rounded-full ${resource.color} mt-2`}></div>
                    <div>
                      <h4 className="font-medium text-gray-900">{resource.name}</h4>
                      <p className="text-sm text-gray-600">{resource.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Daily Schedule */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-100 text-green-600 rounded-lg">
              <Users className="h-5 w-5" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900">{schedule.title}</h3>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {schedule.sessions.map((session, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-semibold text-gray-900 mb-3">{session.time}</h4>
                <ul className="space-y-2">
                  {session.activities.map((activity, actIndex) => (
                    <li key={actIndex} className="text-sm text-gray-600 flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2"></div>
                      {activity}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center mt-12">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-4">Ready to Start Your Journey?</h3>
            <p className="text-lg mb-6 text-blue-100">
              Begin with Week 1 and follow this sequence. Each week builds upon the previous one. 
              Don't skip ahead - the foundation is crucial for advanced topics.
            </p>
            <div className="inline-flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              <Award className="h-5 w-5" />
              Start Learning Today
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ResourcesSection;