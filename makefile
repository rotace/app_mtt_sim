# makefile



all:

run:
	python3 main.py

anal:
	python3 analyzers.py

test:
	python3 -m unittest

clean:
	rm -rf octave-workspace
	rm -rf *.csv *.db *.png