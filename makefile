# makefile



all:

run:
	python3 main.py

anal:
	python3 analyzers.py

test:
	python3 -m unittest

clean:
	rm -rf *.csv *.db octave-workspace