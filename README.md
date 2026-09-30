\# VeriSight



\## AI-Generated and Spam Review Detection System



VeriSight is a machine-learning-based review integrity system designed to analyze online reviews from two complementary perspectives:



1\. Whether a review is human-written or AI-generated.

2\. Whether a review is genuine or spam.



The two detection stages are combined to produce four possible outcomes:



\- Human Genuine

\- Human Spam

\- AI Genuine

\- AI Spam



The system also assigns a risk level to help identify reviews requiring further attention.



\---



\## Project Architecture



```text

&#x20;                   Review

&#x20;                      |

&#x20;                      v

&#x20;             +----------------+

&#x20;             |    Stage 1     |

&#x20;             | Human vs AI    |

&#x20;             +-------+--------+

&#x20;                     |

&#x20;                     v

&#x20;             +----------------+

&#x20;             |    Stage 2     |

&#x20;             | Spam Detection |

&#x20;             +-------+--------+

&#x20;                     |

&#x20;                     v

&#x20;             +----------------+

&#x20;             |  Risk Engine   |

&#x20;             +-------+--------+

&#x20;                     |

&#x20;         +-----------+-----------+

&#x20;         |           |           |

&#x20;         v           v           v

&#x20;        LOW       MEDIUM       HIGH

