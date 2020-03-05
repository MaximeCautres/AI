all:	pdf

tex:
	@latex slide.tex

fast:
	@pdflatex slide.tex

pdf:
	@pdflatex slide.tex
	@pdflatex  slide.tex

read:
	@evince slide.pdf &

clean:
	@echo "Cleaning files..."
	@rm -f *.out	
	@rm -f *.aux
	@rm -f *.bbl
	@rm -f *.blg
	@rm -f *.dvi
	@rm -f *.log
	@rm -f *.out
	@rm -f *.ps
	@rm -f *.toc
	@rm -f *.nav
	@rm -f *.snm
	@rm -f *.synctex.gz
	@rm -f *.backup
	@rm -f *~

