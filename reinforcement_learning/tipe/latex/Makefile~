all:	pdf

tex:
	@latex MCOT.tex


pdf:
	@pdflatex MCOT.tex
	#@bibtex MCOT
	@pdflatex MCOT.tex
	@pdflatex  --synctex=1  MCOT.tex
	@evince MCOT.pdf

clean:
	@echo "Cleaning files..."
	@rm -f *.aux
	@rm -f *.bbl
	@rm -f *.blg
	@rm -f *.dvi
	@rm -f *.log
	@rm -f *.out
	@rm -f *.ps
	@rm -f *.toc
	@rm -f *.synctex.gz
	@rm -f *.backup
	@rm -f *~

