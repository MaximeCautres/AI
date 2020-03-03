all:	pdf

tex:
	@latex récap.tex

fast:
	@pdflatex récap.tex

pdf:
	@pdflatex récap.tex
	@pdflatex  --synctex=1  récap.tex

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

