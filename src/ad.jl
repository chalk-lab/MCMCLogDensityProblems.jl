# Wrappers for gradient, hessian and Hessian-vector product functions

function gen_grad(func, x::AbstractArray; adtype = AutoReverseDiff())
    function retval_sum(x)
        retval = func(x)
        return sum(retval)
    end
    prep = DI.prepare_gradient(retval_sum, adtype, x)
    function grad(x)
        return func(x), DI.gradient(retval_sum, prep, adtype, x)
    end
    return grad
end

function gen_hess(func, x::AbstractArray; adtype = AutoReverseDiff())
    prep = DI.prepare_hessian(func, adtype, x)
    function hess(x::AbstractArray)
        DI.value_gradient_and_hessian(func, prep, adtype, x)
    end
    return hess
end

function gen_hvp(func, x::AbstractArray, v; adtype = AutoReverseDiff())
    prep = DI.prepare_hvp(func, adtype, x, (v,))
    function hvp(x::AbstractArray, v)
        first(DI.hvp(func, prep, adtype, x, (v,)))
    end
    return hvp
end

# Helper function for compiling gradient and hessian functions
gen_logpdf(target) = x -> logpdf(target, x)
gen_logpdf_grad(target, x) = gen_grad(gen_logpdf(target), x)
    logpdf_grad(target, x) = gen_logpdf_grad(target, x)(x)
gen_logpdf_hess(target, x) = gen_hess(gen_logpdf(target), x)
    logpdf_hess(target, x) = gen_logpdf_hess(target, x)(x)
gen_logpdf_hvp(target, x, v) = gen_hvp(gen_logpdf(target), x, v)
    logpdf_hvp(target, x, v) = gen_logpdf_hvp(target, x, v)(x, v)
