
# Tracker concept

Goal:
- Obtain position measurements on targets in FOV

Strategy A:
- Fixed PFA
- Given current target positions
- Window of motion around position

Strategy B:
- Given current target positions
- Feedback on PFA to maximize probability of detection of target
  - Error signal is a function of whether target was detected
  - Cap out at a min PFA

