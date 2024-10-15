CXX = g++-14
CXXFLAGS = -fopenmp -std=c++2a -O3 -I/opt/homebrew/Cellar/gsl/2.8/include
LDFLAGS = -L/opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas

# Target executable
TARGET = Lebwohl-Lasher

# Source files
SRCS = LebwohlLasher.cpp 

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean