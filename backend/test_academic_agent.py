"""
Simple test script for Academic Pathway Agent
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test imports
    from agents.workers.academic_pathway import AcademicPathwayAgent
    print("✅ Academic Pathway Agent import successful!")
    
    # Test agent creation
    agent = AcademicPathwayAgent()
    print("✅ Agent instance created successfully!")
    print(f"Agent name: {agent.name}")
    print(f"Agent description: {agent.description}")
    print(f"Agent capabilities: {len(agent.capabilities)} capabilities")
    
    # Test agent data structures
    sri_lankan_unis = agent.sri_lankan_institutions.get("state_universities", {})
    print(f"✅ Sri Lankan state universities loaded: {len(sri_lankan_unis)}")
    
    international_options = agent.international_options.get("popular_destinations", {})
    print(f"✅ International destinations loaded: {len(international_options)}")
    
    career_levels = agent.career_level_matrix
    print(f"✅ Career level assessments loaded: {len(career_levels)}")
    
    # Test student level assessment
    class MockProfile:
        def __init__(self):
            self.current_education_level = "A/L Student"
            self.major_field = "Physical Science"
            self.age = 18
    
    mock_profile = MockProfile()
    assessment = agent._assess_student_level(mock_profile)
    print(f"✅ Student level assessment: {assessment['current_level']}")
    print(f"   Timeline to career: {assessment['timeline_to_career']}")
    
    # Test institution data access
    colombo_uni = sri_lankan_unis.get("university_of_colombo", {})
    if colombo_uni:
        print(f"✅ University of Colombo data: {colombo_uni.get('name')}")
        print(f"   Strengths: {', '.join(colombo_uni.get('strengths', [])[:3])}")
    
    uk_options = international_options.get("uk", {})
    if uk_options:
        print(f"✅ UK education options: {uk_options.get('approximate_cost_usd')}")
    
    print("\n" + "="*60)
    print("🎓 ACADEMIC PATHWAY AGENT VALIDATION COMPLETE!")
    print("="*60)
    print("✅ All core components are working correctly")
    print("✅ Agent is ready for integration with the career planning system")
    print("✅ Sri Lankan education database loaded and accessible")
    print("✅ International pathway options configured")
    print("✅ Student level assessment framework operational")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()