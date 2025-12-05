import { useState, useEffect } from 'react';
import { BookOpen, Zap, ArrowLeft, CheckCircle, TrendingUp, Loader } from 'lucide-react';
import Footer from '../../components/Footer';

const ACADEMIC_DATA = {
  title: "Software Engineer",
  pathwayTitle: "Academic Pathway",
  description: "Build a strong foundation in computer science and software development",
  courses: [
    {
      id: 1,
      name: "Bachelor's Degree in Computer Science",
      duration: "4 years",
      details: "Core programming, algorithms, data structures, system design"
    },
    {
      id: 2,
      name: "Web Development Specialization",
      duration: "6 months",
      details: "Full-stack development, React, Node.js, databases"
    },
    {
      id: 3,
      name: "Advanced Algorithms & Data Structures",
      duration: "3 months",
      details: "Graph theory, dynamic programming, optimization techniques"
    },
    {
      id: 4,
      name: "Cloud Computing & DevOps",
      duration: "2 months",
      details: "AWS, Docker, Kubernetes, CI/CD pipelines"
    }
  ]
};

const SKILLS_DATA = {
  title: "Software Engineer",
  pathwayTitle: "Required Skills",
  description: "Develop these essential skills to excel in your career",
  skillGroups: [
    {
      id: 1,
      category: "Technical Skills",
      items: [
        "Programming Languages (Python, JavaScript, Java)",
        "Web Development (HTML, CSS, JavaScript)",
        "Backend Development (Node.js, Django, Spring)",
        "Database Design (SQL, MongoDB)",
        "Version Control (Git, GitHub)"
      ]
    },
    {
      id: 2,
      category: "Professional Skills",
      items: [
        "Problem Solving & Critical Thinking",
        "Team Collaboration & Communication",
        "Project Management & Agile Methodologies",
        "System Design & Architecture",
        "Documentation & Code Quality"
      ]
    },
    {
      id: 3,
      category: "Soft Skills",
      items: [
        "Adaptability & Continuous Learning",
        "Time Management",
        "Leadership & Mentorship",
        "Customer Focus & Empathy",
        "Creativity & Innovation"
      ]
    }
  ]
};

export default function CareerPathwayApp() {
  const [currentView, setCurrentView] = useState<'selection' | 'details'>('selection');
  const [selectedPath, setSelectedPath] = useState<'academic' | 'skills' | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAcademicPathway = async () => {
    setLoading(true);
    setTimeout(() => {
      setSelectedPath('academic');
      setCurrentView('details');
      setLoading(false);
    }, 500);
  };

  const handleSkills = async () => {
    setLoading(true);
    setTimeout(() => {
      setSelectedPath('skills');
      setCurrentView('details');
      setLoading(false);
    }, 500);
  };

  const handleBack = () => {
    setCurrentView('selection');
    setSelectedPath(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50">
      {currentView === 'selection' ? (
        <SelectionView 
          onAcademic={handleAcademicPathway} 
          onSkills={handleSkills}
          loading={loading}
        />
      ) : (
        <DetailsView 
          selectedPath={selectedPath} 
          onBack={handleBack}
          academicData={ACADEMIC_DATA}
          skillsData={SKILLS_DATA}
        />
      )}
      <Footer />
    </div>
  );
}

interface SelectionViewProps {
  onAcademic: () => void;
  onSkills: () => void;
  loading: boolean;
}

function SelectionView({ onAcademic, onSkills, loading }: SelectionViewProps) {
  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-100px)] px-4 py-12">
      <div className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-4 leading-tight">
            Explore Your Predicted Career
          </h1>
          <TypewriterText 
            text="Choose how you'd like to view your career recommendations..." 
            className="text-4xl md:text-3xl font-bold text-teal-500 mb-6"
          />
        </div>

        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SelectionButton
              icon={<BookOpen size={32} />}
              title="Academic Pathway"
              description="Explore courses and qualifications needed for your career"
              onClick={onAcademic}
              loading={loading}
              colorClass="teal"
            />
            <SelectionButton
              icon={<Zap size={32} />}
              title="Required Skills"
              description="Discover the key skills you need to develop"
              onClick={onSkills}
              loading={loading}
              colorClass="cyan"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

interface SelectionButtonProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
  loading: boolean;
  colorClass: 'teal' | 'cyan';
}

function SelectionButton({ icon, title, description, onClick, loading, colorClass }: SelectionButtonProps) {
  const borderColor = colorClass === 'teal' ? 'border-teal-200 hover:border-teal-400 hover:shadow-teal-500/20' : 'border-cyan-200 hover:border-cyan-400 hover:shadow-cyan-500/20';
  const bgColor = colorClass === 'teal' ? 'bg-teal-100 text-teal-600 group-hover:bg-teal-200' : 'bg-cyan-100 text-cyan-600 group-hover:bg-cyan-200';
  const gradColor = colorClass === 'teal' ? 'from-teal-400/10 to-cyan-400/10' : 'from-cyan-400/10 to-teal-400/10';

  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={`group relative overflow-hidden rounded-2xl p-8 transition-all duration-300 border-2 ${borderColor} bg-white shadow-lg hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed`}
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${gradColor} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
      
      <div className="relative z-10 flex flex-col items-center text-center space-y-3">
        <div className={`p-3 rounded-full ${bgColor} transition-all duration-300`}>
          {loading ? <Loader size={32} className="animate-spin" /> : icon}
        </div>
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-1">{title}</h4>
          <p className="text-sm text-gray-600">{description}</p>
        </div>
      </div>
    </button>
  );
}

interface FeatureCardProps {
  title: string;
  color: string;
}

function FeatureCard({ title, color }: FeatureCardProps) {
  return (
    <div className="bg-white/60 backdrop-blur rounded-2xl p-6 text-center border border-teal-100">
      <p className="text-sm text-gray-600">
        <span className="font-semibold text-teal-600">{title.split(' ')[0]}</span> {title.split(' ').slice(1).join(' ')}
      </p>
    </div>
  );
}

interface DetailsViewProps {
  selectedPath: 'academic' | 'skills' | null;
  onBack: () => void;
  academicData: typeof ACADEMIC_DATA;
  skillsData: typeof SKILLS_DATA;
}

function DetailsView({ selectedPath, onBack, academicData, skillsData }: DetailsViewProps) {
  const data = selectedPath === 'academic' ? academicData : skillsData;

  return (
    <div className="min-h-[calc(100vh-100px)] px-4 md:px-12 py-12">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onBack}
            className="flex items-center space-x-2 text-teal-600 hover:text-teal-700 transition font-semibold"
          >
            <ArrowLeft size={20} />
            <span>Back</span>
          </button>
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900">{data.title}</h1>
          <div className="w-20"></div>
        </div>

        <div className="bg-white/80 backdrop-blur rounded-3xl shadow-xl p-8 md:p-12 border border-teal-100 mb-8">
          <SectionHeader 
            title={data.pathwayTitle} 
            selectedPath={selectedPath}
            description={data.description}
          />

          {selectedPath === 'academic' && 'courses' in academicData ? (
            <AcademicPathwayComponent courses={academicData.courses} />
          ) : selectedPath === 'skills' && 'skillGroups' in skillsData ? (
            <RequiredSkillsComponent skillGroups={skillsData.skillGroups} />
          ) : null}
        </div>

        <CTASection />
      </div>
    </div>
  );
}

interface SectionHeaderProps {
  title: string;
  selectedPath: 'academic' | 'skills' | null;
  description: string;
}

function SectionHeader({ title, selectedPath, description }: SectionHeaderProps) {
  return (
    <div className="mb-10">
      <div className="flex items-center space-x-3 mb-6">
        {selectedPath === 'academic' ? (
          <BookOpen className="text-teal-500" size={32} />
        ) : (
          <Zap className="text-cyan-500" size={32} />
        )}
        <h2 className="text-3xl font-bold text-gray-900">{title}</h2>
      </div>
      <p className="text-gray-600 text-lg">{description}</p>
    </div>
  );
}

interface AcademicPathwayComponentProps {
  courses: typeof ACADEMIC_DATA.courses;
}

function AcademicPathwayComponent({ courses }: AcademicPathwayComponentProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold text-gray-900 mb-6">Recommended Courses & Programs</h3>
      {courses.map((course, idx) => (
        <CourseCard key={course.id} course={course} index={idx} total={courses.length} />
      ))}
    </div>
  );
}

interface CourseCardProps {
  course: typeof ACADEMIC_DATA.courses[0];
  index: number;
  total: number;
}

function CourseCard({ course, index, total }: CourseCardProps) {
  return (
    <div className="border-2 border-teal-200 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 hover:border-teal-400">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="text-lg font-bold text-gray-900 flex items-center space-x-2">
            <CheckCircle size={20} className="text-teal-500" />
            <span>{course.name}</span>
          </h4>
          <p className="text-sm text-teal-600 font-semibold mt-1">Duration: {course.duration}</p>
        </div>
        <span className="px-4 py-2 bg-teal-100 text-teal-700 rounded-full text-sm font-semibold">
          {index + 1}/{total}
        </span>
      </div>
      <p className="text-gray-600">{course.details}</p>
    </div>
  );
}

interface RequiredSkillsComponentProps {
  skillGroups: typeof SKILLS_DATA.skillGroups;
}

function RequiredSkillsComponent({ skillGroups }: RequiredSkillsComponentProps) {
  return (
    <div className="space-y-8">
      {skillGroups.map((skillGroup) => (
        <SkillGroupCard key={skillGroup.id} skillGroup={skillGroup} />
      ))}
    </div>
  );
}

interface SkillGroupCardProps {
  skillGroup: typeof SKILLS_DATA.skillGroups[0];
}

function SkillGroupCard({ skillGroup }: SkillGroupCardProps) {
  return (
    <div className="border-2 border-cyan-200 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 hover:border-cyan-400">
      <h4 className="text-lg font-bold text-gray-900 flex items-center space-x-2 mb-4">
        <TrendingUp size={24} className="text-cyan-500" />
        <span>{skillGroup.category}</span>
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {skillGroup.items.map((skill: string, i: number) => (
          <SkillItem key={i} skill={skill} />
        ))}
      </div>
    </div>
  );
}

interface SkillItemProps {
  skill: string;
}

function SkillItem({ skill }: SkillItemProps) {
  return (
    <div className="flex items-start space-x-3 bg-cyan-50 rounded-lg p-3">
      <CheckCircle size={18} className="text-cyan-500 flex-shrink-0 mt-0.5" />
      <span className="text-gray-700">{skill}</span>
    </div>
  );
}

function CTASection() {
  return (
    <div className="bg-gradient-to-r from-teal-500 to-cyan-500 rounded-2xl p-8 text-center text-white">
      <h3 className="text-2xl font-bold mb-2">Ready to get started?</h3>
      <button className="bg-white text-teal-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition">
        Explore Resources
      </button>
    </div>
  );
}

interface TypewriterTextProps {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
}

function TypewriterText({ text, speed = 50, delay = 100, className = "text-gray-600 text-lg" }: TypewriterTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    let currentIndex = 0;
    let timeoutId: NodeJS.Timeout;

    const startTyping = () => {
      const type = () => {
        if (currentIndex < text.length) {
          setDisplayedText(text.substring(0, currentIndex + 1));
          currentIndex++;
          timeoutId = setTimeout(type, speed);
        } else {
          setIsComplete(true);
        }
      };

      timeoutId = setTimeout(type, delay);
    };

    startTyping();

    return () => clearTimeout(timeoutId);
  }, [text, speed, delay]);

  return (
    <p className={className}>
      {displayedText}
      {!isComplete && <span className="animate-pulse">|</span>}
    </p>
  );
}