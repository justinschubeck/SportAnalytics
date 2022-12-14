import numpy as np

# ---------------- 3D & 2D Defined Coordinates ------------- # 
coord_system_3D = np.array([[0   , 0  , 0  ],
                            [100 , 0  , 0  ],
                            [100 , 191, 0  ],
                            [100 , 322, 0  ],
                            [0   , 322, 0  ],
                            [-100, 322, 0  ],
                            [-100, 191, 0  ],
                            [-100, 0  , 0  ],
                            [-40 , 68 , 167],
                            [40  , 68 , 167],
                            [0   , 0  , 145],
                            [0   , 101, 170],
                            [0   , 68 , 209]], dtype=np.float32)
                    
coord_system_2D_L = np.array([[220, 483],
                              [264, 455],
                              [531, 458],
                              [721, 462],
                              [767, 487],
                              [848, 535],
                              [565, 532],
                              [158, 524],
                              [338, 192],
                              [358, 206],
                              [227, 244],
                              [405, 201],
                              [346, 134]], dtype=np.float32)
coord_system_2D_C = np.array([[469, 377],
                              [592, 376],
                              [633, 420],
                              [685, 477],
                              [462, 478],
                              [228, 479],
                              [293, 423],
                              [348, 379],
                              [414, 172],
                              [514, 171],
                              [468, 197],
                              [462, 159],
                              [464, 114]], dtype=np.float32)
coord_system_2D_R = np.array([[715, 386],
                              [825, 403],
                              [596, 434],
                              [371, 464],
                              [312, 431],
                              [263, 407],
                              [435, 389],
                              [629, 371],
                              [610, 174],
                              [676, 164],
                              [715, 200],
                              [602, 162],
                              [641, 109]], dtype=np.float32)
# ------------- END: 3D & 2D Defined Coordinates ----------- #