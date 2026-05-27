'use client';

import { useState } from 'react';
import { Phone, MapPin, Clock, Shield, Heart, Users, AlertTriangle, ExternalLink, Search } from 'lucide-react';

interface Professional {
  id: string;
  name: string;
  title: string;
  specialization: string[];
  location: string;
  availability: string;
  contact: string;
  rating: number;
  description: string;
  languages: string[];
  insurance: string[];
  type: 'psychologist' | 'psychiatrist' | 'therapist' | 'counselor';
}

interface CrisisResource {
  id: string;
  name: string;
  description: string;
  phone: string;
  available: string;
  type: 'crisis' | 'general' | 'specialized';
}

export default function ProfessionalReferral() {
  const [searchLocation, setSearchLocation] = useState('');
  const [selectedSpecialization, setSelectedSpecialization] = useState('');
  const [showCrisisResources, setShowCrisisResources] = useState(false);

  const specializations = [
    'Anxiety Disorders',
    'Depression',
    'Trauma & PTSD',
    'Relationship Issues',
    'Substance Abuse',
    'Eating Disorders',
    'Bipolar Disorder',
    'OCD',
    'Grief & Loss',
    'ADHD',
    'Sleep Disorders',
    'General Counseling'
  ];

  const crisisResources: CrisisResource[] = [
    {
      id: 'tele-manas',
      name: 'Tele-MANAS (National Mental Health Helpline)',
      description: 'Government of India mental health helpline offering counseling and crisis support in multiple Indian languages.',
      phone: '14416 / 1-800-91-4416',
      available: '24/7',
      type: 'crisis'
    },
    {
      id: 'kiran-helpline',
      name: 'KIRAN Mental Health Rehabilitation Helpline',
      description: 'MoSJE (Govt. of India) helpline for mental health rehabilitation, emotional support, and referrals.',
      phone: '1800-599-0019',
      available: '24/7',
      type: 'crisis'
    },
    {
      id: 'erss',
      name: 'Emergency Response Support System (ERSS)',
      description: 'National emergency number for immediate assistance, including suicide risk and mental health emergencies.',
      phone: '112',
      available: '24/7',
      type: 'specialized'
    },
    {
      id: 'cybercrime',
      name: 'National Cybercrime Helpline',
      description: 'Report online harassment, cyberstalking, and cybercrimes. Government of India helpline.',
      phone: '1930',
      available: '24/7',
      type: 'general'
    },
    {
      id: 'childline',
      name: 'CHILDLINE (Children in Distress)',
      description: 'Government-supported helpline for children needing support, including mental health crises.',
      phone: '1098',
      available: '24/7',
      type: 'general'
    },
    {
      id: 'women-helpline',
      name: 'Women Helpline (National)',
      description: 'Government helpline for women’s safety and support, including psychosocial assistance.',
      phone: '181',
      available: '24/7',
      type: 'general'
    }
  ];

  // Mock data for demonstration - in a real app, this would come from an API
  const professionals: Professional[] = [
    {
      id: '1',
      name: 'Dr. Priya Sharma',
      title: 'Clinical Psychologist',
      specialization: ['Anxiety Disorders', 'Depression', 'Trauma & PTSD'],
      location: 'Mumbai, Maharashtra',
      availability: 'Monday-Friday, 9AM-6PM',
      contact: '+91-22-2640-1234',
      rating: 4.8,
      description: 'Specializes in cognitive-behavioral therapy and trauma-informed care with 12+ years of experience. Expert in treating anxiety and depression in Indian cultural context.',
      languages: ['Hindi', 'Marathi', 'English', 'Gujarati'],
      insurance: ['ESI', 'CGHS', 'Private Insurance'],
      type: 'psychologist'
    },
    {
      id: '2',
      name: 'Dr. Rajesh Patil',
      title: 'Psychiatrist',
      specialization: ['Bipolar Disorder', 'ADHD', 'Sleep Disorders'],
      location: 'Pune, Maharashtra',
      availability: 'Tuesday-Saturday, 8AM-7PM',
      contact: '+91-20-2553-9876',
      rating: 4.9,
      description: 'Board-certified psychiatrist specializing in mood disorders and medication management. Experienced in treating adolescents and adults.',
      languages: ['Marathi', 'Hindi', 'English'],
      insurance: ['ESI', 'CGHS', 'Private Insurance', 'Ayushman Bharat'],
      type: 'psychiatrist'
    },
    {
      id: '3',
      name: 'Dr. Anjali Deshmukh',
      title: 'Marriage and Family Therapist',
      specialization: ['Relationship Issues', 'Family Therapy', 'Grief & Loss'],
      location: 'Nagpur, Maharashtra',
      availability: 'Monday-Thursday, 10AM-8PM',
      contact: '+91-712-234-5678',
      rating: 4.7,
      description: 'Experienced therapist focusing on family dynamics and relationship counseling. Specializes in Indian family systems and cultural issues.',
      languages: ['Marathi', 'Hindi', 'English'],
      insurance: ['ESI', 'CGHS', 'Private Insurance'],
      type: 'therapist'
    },
    {
      id: '4',
      name: 'Dr. Vikram Joshi',
      title: 'Clinical Psychologist',
      specialization: ['OCD', 'Eating Disorders', 'General Counseling'],
      location: 'Thane, Maharashtra',
      availability: 'Monday-Friday, 9AM-5PM',
      contact: '+91-22-2534-7890',
      rating: 4.6,
      description: 'Specializes in obsessive-compulsive disorders and eating disorders. Provides evidence-based therapy with cultural sensitivity.',
      languages: ['Marathi', 'Hindi', 'English'],
      insurance: ['ESI', 'CGHS', 'Private Insurance'],
      type: 'psychologist'
    },
    {
      id: '5',
      name: 'Dr. Meera Kulkarni',
      title: 'Psychiatrist',
      specialization: ['Depression', 'Anxiety Disorders', 'Substance Abuse'],
      location: 'Nashik, Maharashtra',
      availability: 'Monday-Saturday, 9AM-6PM',
      contact: '+91-253-245-6789',
      rating: 4.8,
      description: 'Expert in treating depression and anxiety with medication management. Specializes in substance abuse treatment and rehabilitation.',
      languages: ['Marathi', 'Hindi', 'English'],
      insurance: ['ESI', 'CGHS', 'Private Insurance', 'Ayushman Bharat'],
      type: 'psychiatrist'
    },
    {
      id: '6',
      name: 'Dr. Sunil Agarwal',
      title: 'Counselor',
      specialization: ['Grief & Loss', 'Relationship Issues', 'General Counseling'],
      location: 'Aurangabad, Maharashtra',
      availability: 'Monday-Friday, 10AM-7PM',
      contact: '+91-240-234-5678',
      rating: 4.5,
      description: 'Experienced counselor specializing in grief counseling and relationship issues. Provides compassionate support for life transitions.',
      languages: ['Marathi', 'Hindi', 'English'],
      insurance: ['ESI', 'CGHS', 'Private Insurance'],
      type: 'counselor'
    }
  ];

  const filteredProfessionals = professionals.filter(professional => {
    const matchesLocation = !searchLocation || 
      professional.location.toLowerCase().includes(searchLocation.toLowerCase());
    const matchesSpecialization = !selectedSpecialization || 
      professional.specialization.includes(selectedSpecialization);
    return matchesLocation && matchesSpecialization;
  });

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'psychologist': return Users;
      case 'psychiatrist': return Heart;
      case 'therapist': return Shield;
      case 'counselor': return Heart;
      default: return Users;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'psychologist': return 'text-primary';
      case 'psychiatrist': return 'text-primary';
      case 'therapist': return 'text-primary';
      case 'counselor': return 'text-primary';
      default: return 'text-slate-600 dark:text-slate-400';
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          Professional Support
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-400">
          Connect with qualified mental health professionals in your area
        </p>
      </div>

      {/* Crisis Resources */}
      <div className="mb-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Crisis Support Resources
          </h2>
          <button
            onClick={() => setShowCrisisResources(!showCrisisResources)}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <AlertTriangle className="h-4 w-4" />
            <span>{showCrisisResources ? 'Hide' : 'Show'} Crisis Resources</span>
          </button>
        </div>

        {showCrisisResources && (
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {crisisResources.map((resource) => (
              <div
                key={resource.id}
                className={`p-6 rounded-xl border-2 ${
                  resource.type === 'crisis'
                    ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                    : resource.type === 'specialized'
                    ? 'bg-primary/10 border-primary/20'
                    : 'bg-success/10 border-success/20'
                }`}
              >
                <div className="flex items-start space-x-4">
                  <div className={`p-3 rounded-lg ${
                    resource.type === 'crisis'
                      ? 'bg-red-100 dark:bg-red-900'
                      : resource.type === 'specialized'
                      ? 'bg-primary/20'
                      : 'bg-success/20'
                  }`}>
                    <Phone className={`h-6 w-6 ${
                      resource.type === 'crisis'
                        ? 'text-red-600 dark:text-red-400'
                        : resource.type === 'specialized'
                        ? 'text-primary'
                        : 'text-success'
                    }`} />
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                      {resource.name}
                    </h3>
                    
                    <p className="text-slate-600 dark:text-slate-400 mb-4">
                      {resource.description}
                    </p>
                    
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Phone className="h-4 w-4 text-slate-500" />
                        <span className="font-mono text-lg font-semibold text-slate-900 dark:text-slate-100">
                          {resource.phone}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Clock className="h-4 w-4 text-slate-500" />
                        <span className="text-sm text-slate-600 dark:text-slate-400">
                          Available: {resource.available}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Search and Filters */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 mb-8">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
          Find a Mental Health Professional
        </h2>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Location
            </label>
            <div className="relative">
              <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={searchLocation}
                onChange={(e) => setSearchLocation(e.target.value)}
                placeholder="Enter city, state, or pincode (e.g., Mumbai, Pune, Maharashtra)"
                className="w-full pl-10 pr-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary dark:bg-slate-700 dark:text-slate-100"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Specialization
            </label>
            <select
              value={selectedSpecialization}
              onChange={(e) => setSelectedSpecialization(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary dark:bg-slate-700 dark:text-slate-100"
            >
              <option value="">All Specializations</option>
              {specializations.map((spec) => (
                <option key={spec} value={spec}>{spec}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Professionals List */}
      <div className="space-y-6">
        {filteredProfessionals.length === 0 ? (
          <div className="text-center py-12">
            <Search className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
              No professionals found
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Try adjusting your search criteria or contact us for assistance.
            </p>
          </div>
        ) : (
          filteredProfessionals.map((professional) => {
            const TypeIcon = getTypeIcon(professional.type);
            return (
              <div
                key={professional.id}
                className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-lg bg-slate-100 dark:bg-slate-700 ${getTypeColor(professional.type)}`}>
                      <TypeIcon className="h-6 w-6" />
                    </div>
                    
                    <div>
                      <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-1">
                        {professional.name}
                      </h3>
                      <p className="text-slate-600 dark:text-slate-400 mb-2">
                        {professional.title}
                      </p>
                      <div className="flex items-center space-x-4 text-sm text-slate-500 dark:text-slate-400">
                        <div className="flex items-center space-x-1">
                          <MapPin className="h-4 w-4" />
                          <span>{professional.location}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Clock className="h-4 w-4" />
                          <span>{professional.availability}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                      ⭐ {professional.rating}
                    </div>
                    <button className="mt-2 bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                      Contact
                    </button>
                  </div>
                </div>
                
                <p className="text-slate-600 dark:text-slate-400 mb-4">
                  {professional.description}
                </p>
                
                <div className="grid md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <h4 className="font-medium text-slate-900 dark:text-slate-100 mb-2">Specializations</h4>
                    <div className="flex flex-wrap gap-1">
                      {professional.specialization.map((spec, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-primary/20 text-primary text-xs rounded-full"
                        >
                          {spec}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-slate-900 dark:text-slate-100 mb-2">Languages</h4>
                    <div className="flex flex-wrap gap-1">
                      {professional.languages.map((lang, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-success/20 text-success text-xs rounded-full"
                        >
                          {lang}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-slate-900 dark:text-slate-100 mb-2">Insurance</h4>
                    <div className="flex flex-wrap gap-1">
                      {professional.insurance.map((ins, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-primary/20 text-primary text-xs rounded-full"
                        >
                          {ins}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between pt-4 border-t border-slate-200 dark:border-slate-700">
                  <div className="flex items-center space-x-2 text-slate-600 dark:text-slate-400">
                    <Phone className="h-4 w-4" />
                    <span className="font-mono">{professional.contact}</span>
                  </div>
                  
                  <button className="flex items-center space-x-2 text-primary hover:text-primary/80 font-medium">
                    <span>View Profile</span>
                    <ExternalLink className="h-4 w-4" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Additional Resources */}
      <div className="mt-12 bg-primary/10 rounded-2xl p-8 border border-primary/20">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6 text-center">
          Additional Resources
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Finding the Right Professional
            </h3>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400">
              <li>• Consider your specific needs and preferences</li>
              <li>• Check if they accept your insurance (ESI, CGHS, Ayushman Bharat)</li>
              <li>• Verify language preferences (Hindi, Marathi, English)</li>
              <li>• Read reviews and ask for recommendations</li>
              <li>• Schedule a consultation to see if it&apos;s a good fit</li>
              <li>• Don&apos;t hesitate to try different professionals</li>
            </ul>
          </div>
          
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              What to Expect
            </h3>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400">
              <li>• Initial consultation to discuss your needs</li>
              <li>• Development of a treatment plan</li>
              <li>• Regular sessions (weekly or bi-weekly)</li>
              <li>• Progress monitoring and plan adjustments</li>
              <li>• Confidentiality and professional boundaries</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
