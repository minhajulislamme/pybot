#!/usr/bin/env python3

import unittest
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_results.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TestRunner")

def run_tests():
    """Discover and run all tests"""
    try:
        logger.info("Starting test discovery...")
        
        # Discover all tests in the tests directory
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern='test_*.py')
        
        logger.info(f"Found {suite.countTestCases()} test cases")
        
        # Run the tests
        logger.info("Running tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print test summary
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Errors: {len(result.errors)}")
        logger.info(f"Failures: {len(result.failures)}")
        
        if not result.wasSuccessful():
            logger.error("Some tests failed")
            return False
        else:
            logger.info("All tests passed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
