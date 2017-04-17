% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
function make()
try
	% This part is for OCTAVE
	if(exist('OCTAVE_VERSION', 'builtin'))
		mex libsvmread.c
		mex libsvmwrite.c
		mex -I.. train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
		mex -I.. predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.cpp
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.cpp
		mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims train.cpp linear_model_matlab.cpp ../linear.cpp ../tron.cpp ../blas/daxpy.cpp ../blas/ddot.cpp ../blas/dnrm2.cpp ../blas/dscal.cpp
		mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims predict.cpp linear_model_matlab.cpp ../linear.cpp ../tron.cpp ../blas/daxpy.cpp ../blas/ddot.cpp ../blas/dnrm2.cpp ../blas/dscal.cpp
	end
catch err
	fprintf('Error: %s failed (line %d)\n', err.stack(1).file, err.stack(1).line);
	disp(err.message);
	fprintf('=> Please check README for detailed instructions.\n');
end
