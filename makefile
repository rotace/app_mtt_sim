# makefile



all:

run:
	python3 main.py

view:
	python3 viewers.py

test:
	python3 -m unittest

clean:
	rm -rf *.csv