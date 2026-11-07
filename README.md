# Financial_engineering_report
Financial_engineering_report

Noter til hoved artikler: Hoved artiklerne er følgende
Carr, P. and Madan, D. (1999). Option Valuation using the Fast Fourier Transform, Journal of
Computational Finance

Madan, D. and Seneta, E. (1990). The Variance Gamma Model for Share Market Returns,
Journal of Business

Schmelzle, M. (2010). Option Pricing Formulae using Fourier Transform: Theory and
Application, working paper

Carr, P., Geman, H., Madan, D. B., & Yor, M. (2002). The fine structure of asset returns: An
empirical investigation. The Journal of Business.

Noter til Carr, P. and Madan, D. (1999).
### Abstract Noter
The purpose of this paper is to describe a new approach for numerically
determining option values, which is designed to use the FFT to value options
e±ciently. Given any such characteristic function, we develop a simple
analytic expression for the Fourier transform of the option value or its time
value.

### Noter til konklusion 
developed two Fourier transforms in terms of the characteristic function of the log of the terminal stock price. The ¯rst is the Fourier
transform of the modi¯ed call price written as a function of log strike, where
the modi¯cation involves multiplying by an exponential. The second is the   
Fourier transform of the modi¯ed time value, where the modi¯cation involves
multiplying by the hyperbolic sine function.


## Noter til resten af papir
The methods are generally much faster than ¯nite di®erence solutions to partial di®erential equations or integro-di®erential equations, which led Heston to refer to them as closed form solutions.
Assuming that the characteristic function is known analytically, many authors 
Assuming no dividends and constant interest rates r; the initial option value
is then determined as:
C = S pi_1 - Ke^rT pi_2:
Two New Fourier Methods
In this section, we develop analytic expressions for the Fourier transform of
an option price and for the Fourier transform of the time value of an option.
the call pricing
function is not square integrable To obtain a square integrable function, we
consider the modi¯ed call price. mass udregninger 
We now consider the issue of the appropriate choice of the coe±cient alpha.


Fourier Transform of Out-Of-The-Money Option Prices
The last section multiplied call values by an exponential function to obtain
a square integrable function whose Fourier transform is an analytic function
of the characteristic function of the log price. Unfortunately, for very short
maturities, the call value approaches its non-analytic intrinsic value causing
the integrand in the Fourier inversion to be highly oscillatory, and therefore
di±cult to numerically integrate.


Option Pricing Using the FFT
Tror ik det er vigtigt for os.

Mine tanker om papir. Det er lidt sådan bare det vi skal bruge af teknikker til at regne vores 
call priser ud på. mat delen var ikke super overvælende. de brugte meget tid om valg af alpha 
og hvorfor det var vigtigt. værd at tage i mente når vi skal skrive vores papir. 
Closed form solutions til at når vi kender vores char funtion. brude være nemt at kode op.
